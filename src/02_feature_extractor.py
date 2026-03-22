"""Modulo 2 — Feature Extraction (CAL su backbone C2DResNet50) → Embedding.

Carica il backbone C2DResNet50 pre-addestrato con CAL (Clothes-based
Adversarial Loss) su MEVID e calcola un embedding 2048-d per ogni
track estratta dal Modulo 1.

Input:
    Cartella di ROI prodotta dal Modulo 1, organizzata come::

        extracted_rois/<video_name>/Track_XXXX/frame_YYYYYY.jpg

Output:
    Per ogni video → ``embeddings.pt`` con un dizionario
    ``{track_id: Tensor(2048)}`` e un ``embedding_report.json``
    con statistiche aggregate (track processate, tempo, ecc.).

Gli embedding alimentano il Modulo 3 (Cross-Camera Matching)
per il confronto e il clustering delle identità tra telecamere diverse.
"""
