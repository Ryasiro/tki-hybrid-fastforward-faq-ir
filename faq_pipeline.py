"""Antarmuka Explainable IR (Streamlit) — Dataset FAQ & Pedoman Akademik (43 Entitas)."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from faq_core import HybridFAQ, MODEL_NAME

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="FAQ Akademik UNSRI | Hybrid Fast-Forward IR",
    page_icon="🎓",
    layout="wide",
)


@st.cache_data
def load_corpus():
    return json.loads((ROOT / "dataset_pedoman_akademik.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Memuat MiniLM-L12-v2 (CPU) & membangun indeks 43 entitas FAQ...")
def load_engine(negation_safe: bool = False):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    return HybridFAQ(load_corpus(), model, delta=0.035, max_passage_words=120, negation_safe=negation_safe)


corpus = load_corpus()
n_faq = sum(1 for x in corpus if x.get("type") == "faq")
n_pas = sum(1 for x in corpus if x.get("type") == "passage")
domains = sorted({x.get("domain", x.get("category", "")) for x in corpus})

# Header bersih
st.title("🎓 Sistem Temu Kembali FAQ & Pedoman Akademik")
st.caption(
    f"Hybrid BM25 (IDF Positif) + Fast-Forward Dense Index (MiniLM-L12-v2 CPU) • "
    f"Korpus terkurasi: **{len(corpus)} entitas** ({n_faq} FAQ + {n_pas} Pasal Pedoman across {len(domains)} domain)."
)

# Sidebar parameter
with st.sidebar:
    st.header("⚙️ Parameter Pencarian")
    alpha = st.slider(
        "Bobot Interpolasi α (0 = Dense, 1 = BM25)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    top_k = st.slider("Jumlah Hasil (Top-K)", min_value=1, max_value=10, value=5, step=1)
    early = st.checkbox("Safe Early Stopping (batas cosine ≤ 1)", value=True)
    neg_safe = st.checkbox("Mode Negation-Safe (pertahankan kata 'tidak')", value=False)

    st.divider()
    if alpha == 0.0:
        st.info("**Mode: Pure Dense** — 100% semantik MiniLM.")
    elif alpha == 1.0:
        st.warning("**Mode: Pure Sparse BM25** — 100% leksikal; hasil kosong jika tanpa kecocokan kata.")
    else:
        st.success(f"**Mode: Hybrid** — Leksikal {alpha*100:.0f}% | Semantik {(1-alpha)*100:.0f}%")

    st.divider()
    st.markdown("**👥 Kelompok 8 — TKI UNSRI**")
    st.caption(
        "M. Radikal Gymnastiar (09021282328050)\n"
        "M. Ilham Syafik (09021182328015)\n"
        "Muhamad Arya Hatta (09021282328094)\n"
        "Aditya Akmal Agym (09021282328072)"
    )

# Preset kueri cepat yang rapi
presets = {
    "— Ketik pertanyaan sendiri —": "",
    "Parafrasa Login SIAKAD (tanpa overlap)": "tidak ingat kode akses masuk sistem",
    "Parafrasa KTM Hilang → KPM UNSRI": "KTM saya patah dan hilang bagaimana urus gantinya",
    "Parafrasa Lapor Pelecehan → SATGAS PPKPT": "bagaimana cara lapor tindakan pelecehan di lingkungan kampus",
    "Parafrasa Syarat Bahasa Inggris → USEPT": "sertifikat nilai bahasa inggris apa yang wajib buat lulus sarjana",
    "Parafrasa Diskon UKT Skripsi": "tinggal nulis skripsi doang bisa dapat diskon uang kuliah",
    "Parafrasa Sakit saat Ujian": "masuk rumah sakit pas jadwal tes akhir apakah boleh ikut belakangan",
    "Pasal Sanksi Joki / Kecurangan": "ketahuan nyontek atau pakai jasa joki saat tes",
    "Syarat Predikat Cumlaude": "syarat meraih predikat Dengan Pujian Cumlaude tanpa nilai C dan skripsi A",
}

col_sel, col_inp = st.columns([1.2, 2.3])
with col_sel:
    selected = st.selectbox("Contoh kueri pengujian:", list(presets.keys()))
with col_inp:
    query = st.text_input(
        "Masukkan pertanyaan mahasiswa:",
        value=presets[selected],
        placeholder="Contoh: bagaimana cara cicil UKT atau syarat ikut KKN...",
    )

if query.strip():
    try:
        engine = load_engine(negation_safe=neg_safe)
        r = engine.search(query, alpha=alpha, top_k=top_k, early_stopping=early)

        if not r.hits:
            st.warning(
                "Tidak ada kecocokan leksikal pada mode BM25 murni (α = 1,0). "
                "Turunkan α < 1,0 di sidebar untuk mengaktifkan pencarian semantik."
            )
        else:
            top = r.hits[0]
            if top.source_status == "terverifikasi_pedoman_unsri_2026_2027":
                badge = "📘 Terverifikasi Pedoman UNSRI 2026/2027"
            else:
                badge = "💬 FAQ Operasional Terkurasi"
            st.markdown("---")
            st.subheader("🎯 Jawaban Terbaik (Peringkat #1)")
            with st.container(border=True):
                st.markdown(f"**[{top.id}] {top.title}** &nbsp; `{badge}` &nbsp; `{top.domain}`")
                st.markdown(f"*{top.question}*")
                st.write(top.answer)
                if top.source_reference:
                    st.caption(f"📖 Sumber: {top.source_reference}")
                if top.keywords:
                    st.caption("Kata kunci terkait: " + " • ".join(f"`{k}`" for k in top.keywords))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Skor Hybrid (bukan probabilitas)", f"{top.hybrid:.4f}")
            c2.metric("BM25 Norm / Dense", f"{top.norm_bm25:.3f} / {f'{top.dense:.3f}' if top.dense is not None else '-'}")
            c3.metric("Dense Look-ups", f"{r.evaluated_dense} / {r.total_documents}")
            c4.metric("Latensi Core CPU", f"{r.time_total_core_ms:.2f} ms")

            # Tabel dekomposisi skor (Explainable IR)
            st.subheader(f"📊 Dekomposisi Skor Top-{len(r.hits)} (Explainable IR)")
            rows = [
                {
                    "Rank": f"#{x.rank}",
                    "ID": x.id,
                    "Tipe": x.doc_type.upper(),
                    "Domain": x.domain,
                    "Judul": x.title,
                    "BM25 Raw": round(x.raw_bm25, 4),
                    "BM25 Norm": round(x.norm_bm25, 4),
                    "Dense Cosine": round(x.dense, 4) if x.dense is not None else None,
                    "Skor Hybrid": round(x.hybrid, 4),
                }
                for x in r.hits
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            # Lihat jawaban kandidat lainnya (#2..#K)
            if len(r.hits) > 1:
                with st.expander(f"Lihat rincian jawaban Peringkat #2 – #{len(r.hits)}"):
                    for h in r.hits[1:]:
                        st.markdown(f"**#{h.rank} [{h.id}] {h.title}** (`{h.domain}` • Skor `{h.hybrid:.4f}`)")
                        st.write(h.answer)
                        st.divider()

            st.caption(
                f"Token BM25: `{r.tokens}` | Early stop aktif: `{r.stopped_early}` | "
                f"Query encoder: `{r.time_query_encoder_ms:.2f} ms` | "
                f"Passages: `{engine.raw_passage_count} → {engine.coalesced_count}` (kompresi `{engine.compression_ratio*100:.1f}%`)."
            )
    except Exception as e:
        st.error(f"Pencarian gagal: {e}")
else:
    st.info("Pilih contoh kueri di sebelah kiri atau ketik pertanyaan mahasiswa untuk melihat hasil dan jejak skor.")
    with st.expander(f"📚 Lihat seluruh {len(corpus)} entitas dalam basis pengetahuan"):
        df_all = pd.DataFrame(
            [{"ID": x["id"], "Tipe": x["type"].upper(), "Domain": x["domain"], "Judul": x["title"], "Pertanyaan": x["question"]} for x in corpus]
        )
        st.dataframe(df_all, width="stretch", hide_index=True)
