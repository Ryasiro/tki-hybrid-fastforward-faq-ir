# Sistem Temu Kembali FAQ & Pedoman Akademik UNSRI (43 Entitas Tersinkronisasi)

Paket implementasi, evaluasi reproduksibel, antarmuka Streamlit, dan naskah terverifikasi berbasis **`dataset_pedoman_akademik.json`** / **`dataset_pedoman_akademik.csv`** (43 entitas: **32 terverifikasi langsung terhadap Buku Pedoman Akademik & Kemahasiswaan UNSRI 2026/2027** + **11 kurasi operasional unit terkait**).

## Sinkronisasi Fakta Terhadap Buku Pedoman UNSRI 2026/2027

- `ADM_001`: **KPM** (*Kartu Pengenal Mahasiswa*, Sub-bab 6.1.3 & Lampiran 5) — menangani parafrasa mahasiswa *"KTM hilang"*.
- `ADM_004` vs `PEDOMAN_BAB4_P3`: membedakan secara tegas **Surat Keterangan Pengganti Ijazah/Transkrip** (Sub-bab 6.2.2, Lampiran 15) dari **SKPI** (*Surat Keterangan Pendamping Ijazah*, Sub-bab 5.5).
- `AKD_001`: minimal kehadiran perkuliahan **85%** (Sub-bab 3.2.2, hal. 40).
- `CUTI_001` & `CUTI_002`: **Stop Out (SO)** maksimal **1 semester** untuk D3/S1/S2, syarat minimal 50% SKS, batas pengajuan minggu ke-3, dan **diperhitungkan sebagai masa studi** (Sub-bab 3.4.1, 3.9, 6.1.4).
- `TA_003`: syarat yudisium & skor **USEPT** (*Universitas Sriwijaya English Proficiency Test*, Sub-bab 3.12.1 & 4.2.10).
- `PEDOMAN_BAB5_P1..P2`: evaluasi akhir tahun ke-2 **52 SKS** & IPK ≥ 2,00, masa studi maks 14 semester (Sub-bab 3.8.1).
- `PEDOMAN_BAB7_P1..P2`: pelaporan **SATGAS PPKPT** (Sub-bab 4.4) & predikat **Cumlaude** (Sub-bab 3.12.2 Tabel 3.9: IPK 3,51–4,00, maks 8 semester, tanpa nilai C, skripsi A).

## Cara Menjalankan

```powershell
cd "C:\Users\USER\Documents\tki\Paket_Revisi_FAQ_IR\draft_revisi2_faq43"
pytest -q
python evaluate.py --top-k 5 --warmup 10 --repeats 8
streamlit run faq_pipeline.py
```

## Ringkasan Hasil Empiris (Unseen Test Set, n = 30, CPU)

Hiperparameter dipilih pada **Dev Set (`n = 20`)**: **`α* = 0,1`**.

| Model (`N = 43`) | Hemat Lookup | Recall@5 | MRR@5 | MAP@5 | nDCG@5 | Parafrasa nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 Murni (`α=1,0`) | N/A | 0,9167 | 0,8778 | 0,8639 | 0,8873 | 0,6675 |
| Dense Murni (`α=0,0`) | 0,00% | 0,9000 | 0,8500 | 0,8222 | 0,8544 | 0,6457 |
| Hybrid Default (`α=0,5`, ES) | 3,33% | 0,9500 | 0,8861 | 0,8750 | 0,9023 | — |
| **Hybrid Safe Early Stop (`α*=0,1`)** | **0,00%** | **0,9500** | **0,9278** | **0,9028** | **0,9301** | **0,7957** |

- **Uji otomatis:** **20/20 lulus** (`hasil_pytest.txt`, termasuk regresi faktual Pedoman UNSRI 2026/2027).
- **Early Stopping vs Exhaustive:** **0 selisih peringkat** pada seluruh 30 kueri test (hemat lookup pada dev: `31,40%` di `α=0,7`, `58,95%` di `α=0,8`, `76,40%` di `α=0,9`).
- **Latensi CPU (240 eksekusi pasca 10× warm-up):** core `p50 = 16,80 ms` (p95 `18,66 ms`, ranking saja `0,83 ms`), BM25 murni `0,14 ms`.
