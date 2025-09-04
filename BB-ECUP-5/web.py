
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from starlette.concurrency import run_in_threadpool
from pathlib import Path
from uuid import uuid4
import shutil
import time


from main import final

app = FastAPI()

OUTPUT_DIR = Path("output")          # папка, куда бэкенд кладет результа
IN_DIR = Path("workdir/in")          # куда сохраняем загрузки
IN_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!doctype html>
    <html lang="ru">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Загрузка файла</title>
      <style>
        body { font-family: system-ui, sans-serif; margin: 2rem; }
        .card { max-width: 520px; padding: 1.5rem; border: 1px solid #e5e7eb; border-radius: 12px; }
        .row { margin: .75rem 0; }
        button { padding: .6rem 1rem; border: 0; border-radius: 8px; cursor: pointer; background:#111827; color:#fff; }
      </style>
    </head>
    <body>
      <div class="card">
        <h2>Загрузите файл → Выполнить → Скачать результат</h2>
        <form action="/process" method="post" enctype="multipart/form-data">
          <div class="row"><input type="file" name="file" required /></div>
          <div class="row"><button type="submit">Выполнить</button></div>
        </form>
      </div>
    </body>
    </html>
    """

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    # 1) Сохраняем загруженный файл во временный путь
    temp_path = IN_DIR / f"{uuid4()}_{file.filename}"
    with temp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) Запоминаем состояние папки OUTPUT до вызова
    before = {p.resolve() for p in OUTPUT_DIR.glob("*") if p.is_file()}

    # 3) Вызываем ваш бэкенд (в отдельном потоке, чтобы не блокировать event loop)
    await run_in_threadpool(final, str(temp_path))

    # 4) Ищем новые файлы в OUTPUT (появившиеся после вызова)
    #    Если по каким-то причинам «новых» нет — берем самый свежий по mtime.
    time.sleep(0.05)  # маленькая пауза на запись файла ОС
    candidates = [p for p in OUTPUT_DIR.glob("*") if p.is_file()]
    new_files = [p for p in candidates if p.resolve() not in before]

    if new_files:
        out_file = max(new_files, key=lambda p: p.stat().st_mtime)
    elif candidates:
        out_file = max(candidates, key=lambda p: p.stat().st_mtime)
    else:
        raise HTTPException(status_code=500, detail="Бэкенд не создал выходной файл в ./output")

    # (опционально) можно удалить входной временный файл
    try:
        temp_path.unlink(missing_ok=True)
    except Exception:
        pass

    # 5) Отдаем файл пользователю
    return FileResponse(
        path=str(out_file),
        media_type="application/octet-stream",
        filename=out_file.name,
    )

from fastapi import Query

@app.get("/download")
def download(filename: str = Query(..., description="Имя файла из папки output")):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(
        path=str(file_path),
        media_type="application/octet-stream",
        filename=file_path.name
    )
