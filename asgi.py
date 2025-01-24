from fastapi import FastAPI, HTTPException
import uvicorn
import logging
from format.input import *
from src.parler import ParlerTTS
app = FastAPI()

logging.basicConfig(level=logging.DEBUG)

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/parler")
async def execute_tts(request: inputPayload):
    try:
        logging.info("Richiesta ricevuta per l'endpoint /parler")

        # Creazione del payload
        input_payload = request

        # Inizializza il processo TTS
        logging.info("Inizializzazione del processo TTS")
        tts = ParlerTTS(input_payload)
        # Esegui TTS
        logging.info("Esecuzione del metodo tts.execute()")
        output_json = tts.execute()
        logging.info("Risultato generato")
        return output_json
    except Exception as e:
        logging.error(f"Errore durante l'elaborazione: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080, use_colors=True)