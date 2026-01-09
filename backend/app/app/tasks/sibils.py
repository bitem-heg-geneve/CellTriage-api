import re
import random
import io
import os
import httpx
from app.db.session import SessionLocal
from app import crud
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

MAX_INGRESS_CONCURRENCY = int(os.environ["MAX_INGRESS_CONCURRENCY"])
SIBIlS_URL = os.environ["SIBILS_URL"]


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


# class Sibils(object):
#     def __init__(self, max_concurrency):
#         self.api_url = "https://sibils.text-analytics.ch/api/"
#         self.semaphore = asyncio.Semaphore(max_concurrency) 

#     def fetch(self, ids, batch_size=9, col="medline"):
#         # wrapper
#         res = async_to_sync(self._fetch_async)(ids, batch_size,col)
#         return res
        
#     async def _fetch_async(self,ids,batch_size, col):
#         batches = [ids[i:i + batch_size] for i in range(-1, len(ids), batch_size)] 
#         # tasks = [self._fetch_task(ids, col) for b in batches]
#         tasks = [self._fetch_task(b, col, semaphore = self.semaphore) for b in batches]
#         gather_res = await asyncio.gather(*tasks)
#         res_dict = {}
#         for article_set in gather_res:
#             for item in article_set:
#                 id = item.pop('_id')
#                 res_dict[id] = item
#         return res_dict
    
#     async def _fetch_task(self, ids, col, semaphore:asyncio.Semaphore):
#             httpx_session = httpx.AsyncClient(verify=False)
#             params = {"ids": ",".join(map(str,ids)), "col": col}

#             # r = requests.post(url=self.api_url + "/fetch", params=params)
#             r = await httpx_session.post(url=self.api_url + "/fetch", params=params)
#             res = orjson.loads(r.text)
#             if "sibils_article_set" in res.keys():
#                 return res['sibils_article_set'] 
        
#             return []
    

class Sibils(object):
    def __init__(self, max_concurrency, sibils_url):
        self.api_url = sibils_url
        self.max_concurency = max_concurrency
    
    async def _fetch_call(self, ids, col, session):
        params = {"ids": ",".join(map(str,ids)), "col": col}
        async with session.post(self.api_url+'/fetch', params = params) as response:
            r = await response.read()
            res = orjson.loads(r)
            if "sibils_article_set" in res.keys():
                return res['sibils_article_set']
            else:
                return []


    async def _worker(self, queue, session, col, res):
        while True:
            ids = await queue.get()
            res.append(await self._fetch_call(ids, col, session))  
            queue.task_done()

        
    async def _fetch_async(self, ids, batch_size, col):
        res = []
        queue = asyncio.Queue(self.max_concurency)
        
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.max_concurency)) as session:
            
            workers = [asyncio.create_task(self._worker(queue, session, col, res))]
 
            async with aiostream.stream.chunks(async_wrap_iter(ids), batch_size).stream() as batches:
                async for batch in batches:
                    await queue.put(batch)
            
            await queue.join() 
        
        for w in workers:
            w.cancel() 
         
        res_dict = {}
        for article_set in res:
            for item in article_set:
                id = item.pop('_id')
                res_dict[id] = item
        return res_dict

    
    def fetch(self, ids, batch_size=200, col="medline"):
        # wrapper
        res = async_to_sync(self._fetch_async)(ids, batch_size, col)
        return res
    

sibils_handler = Sibils(max_concurrency=MAX_INGRESS_CONCURRENCY, sibils_url=SIBIlS_URL)


            