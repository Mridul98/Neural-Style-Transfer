from fastapi import FastAPI


app = FastAPI()


@app.post('/nst')
async def submit_work():
    pass

@app.get('/result/{job_id}')
async def get_result(job_id:str):
    pass