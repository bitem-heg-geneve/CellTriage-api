import re
import random
import io
import os
import httpx
import logging
# import requests
import logging as log
from operator import itemgetter
import orjson
import asyncio
from collections import ChainMap
from asgiref.sync import async_to_sync
import aiostream
import aiohttp
import threading

from app.core.config import settings
from app.services.sibils_api import FetchResult, SibilsFetchError, parse_fetch_response

MAX_INGRESS_CONCURRENCY = int(os.environ["MAX_INGRESS_CONCURRENCY"])


def async_wrap_iter(it):
    """Wrap blocking iterator into an asynchronous one"""
    loop = asyncio.get_event_loop()
    q = asyncio.Queue(1)
    exception = None
    _END = object()

    async def yield_queue_items():
        while True:
            next_item = await q.get()
            if next_item is _END:
                break
            yield next_item
        if exception is not None:
            # the iterator has raised, propagate the exception
            raise exception

    def iter_to_queue():
        nonlocal exception
        try:
            for item in it:
                # This runs outside the event loop thread, so we
                # must use thread-safe API to talk to the queue.
                asyncio.run_coroutine_threadsafe(q.put(item), loop).result()
        except Exception as e:
            exception = e
        finally:
            asyncio.run_coroutine_threadsafe(q.put(_END), loop).result()

    threading.Thread(target=iter_to_queue).start()
    return yield_queue_items()


class Sibils(object):
    """SIBiLS fetch client (shared triage guard, §1).

    `fetch` never answers "here are the records I found" alone: it returns a
    FetchResult that says, per requested id, what came back (`found`,
    `missing`), plus the instance's warnings and versions. Transport failures
    (HTTP error, timeout, non-JSON body, `success: false`) are retried
    `settings.SIBILS_RETRIES` times with backoff and then raised as
    SibilsFetchError; the caller records that as text_status=fetch_error.
    """

    def __init__(self, max_concurrency, sibils_url, timeout=None, retries=None):
        self.api_url = sibils_url
        self.max_concurency = max_concurrency
        self.timeout = settings.SIBILS_TIMEOUT if timeout is None else timeout
        self.retries = settings.SIBILS_RETRIES if retries is None else retries

    async def _fetch_call(self, ids, col, session) -> FetchResult:
        """One POST to /fetch. Raises SibilsFetchError unless a parseable 200 came back."""
        url = self.api_url + '/fetch'
        params = {"ids": ",".join(map(str, ids)), "col": col}
        async with session.post(url, params=params) as response:
            if response.status != 200:
                raise SibilsFetchError(f"HTTP {response.status} from {url} (col={col})")
            r = await response.read()
        try:
            res = orjson.loads(r)
        except orjson.JSONDecodeError as exc:
            raise SibilsFetchError(f"non-JSON body from {url} (col={col}): {exc}") from exc
        return parse_fetch_response(res, ids)

    async def _fetch_retrying(self, ids, col, session) -> FetchResult:
        delay = 2.0
        for attempt in range(1, self.retries + 1):
            try:
                return await self._fetch_call(ids, col, session)
            except (SibilsFetchError, aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                if attempt >= self.retries:
                    raise SibilsFetchError(
                        f"{col} fetch of {len(ids)} ids failed after {attempt} attempts: {exc}"
                    ) from exc
                log.warning(
                    "SIBiLS %s fetch attempt %d/%d failed: %s; retrying in %.0fs",
                    col, attempt, self.retries, exc, delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise AssertionError("unreachable")

    async def _worker(self, queue, session, col, res, errors):
        while True:
            ids = await queue.get()
            try:
                res.append(await self._fetch_retrying(ids, col, session))
            except Exception as exc:
                # Keep draining: an unhandled error here would kill the worker
                # and leave queue.join() waiting forever.
                errors.append(exc)
            finally:
                queue.task_done()

    async def _fetch_async(self, ids, batch_size, col) -> FetchResult:
        res = []
        errors = []
        queue = asyncio.Queue(self.max_concurency)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=self.timeout, sock_read=self.timeout)

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.max_concurency), timeout=timeout) as session:

            workers = [asyncio.create_task(self._worker(queue, session, col, res, errors))]

            async with aiostream.stream.chunks(async_wrap_iter(ids), batch_size).stream() as batches:
                async for batch in batches:
                    await queue.put(batch)

            await queue.join()

        for w in workers:
            w.cancel()

        if errors:
            raise SibilsFetchError(
                f"{len(errors)} of {len(res) + len(errors)} {col} batches failed: {errors[0]}"
            ) from errors[0]
        out = FetchResult()
        for batch_result in res:
            out.merge(batch_result)
        return out

    def fetch(self, ids, batch_size=200, col="medline") -> FetchResult:
        """Fetch `ids` from collection `col`. Raises SibilsFetchError after retries."""
        if not ids:
            return FetchResult()
        return async_to_sync(self._fetch_async)(list(ids), batch_size, col)


sibils_handler = Sibils(max_concurrency=MAX_INGRESS_CONCURRENCY, sibils_url=settings.SIBILS_URL)


            