from __future__ import annotations

import os
import json
from datetime import date

TODAY = str(date.today())

DRUGS = [
    # generic, brands, tags
    ("amlodipine", ["Norvasc", "络活喜"], ["hypertension", "ccb"]),
    ("enalapril", ["Vasotec", "依那普利"], ["hypertension", "acei"]),
    ("losartan", ["Cozaar", "氯沙坦"], ["hypertension", "arb"]),
    ("valsartan", ["Diovan", "缬沙坦"], ["hypertension", "arb"]),
    ("metformin", ["Glucophage", "二甲双胍"], ["diabetes", "biguanide"]),
    ("atorvastatin", ["Lipitor", "阿托伐他汀"], ["lipid", "statin"]),
    ("simvastatin", ["Zocor", "辛伐他汀"], ["lipid", "statin"]),
    ("aspirin", ["Bayer", "阿司匹林"], ["antiplatelet", "nsaid"]),
    ("ibuprofen", ["Advil", "布洛芬"], ["nsaid", "pain"]),
    ("warfarin", ["Coumadin", "华法林"], ["anticoagulant"]),
    ("apixaban", ["Eliquis", "阿哌沙班"], ["anticoagulant", "doac"]),
    ("clopidogrel", ["Plavix", "氯吡格雷"], ["antiplatelet"]),
    ("omeprazole", ["Prilosec", "奥美拉唑"], ["ppi", "gi"]),
    ("amoxicillin", ["Amoxil", "阿莫西林"], ["antibiotic", "penicillin"]),
    ("azithromycin", ["Zithromax", "阿奇霉素"], ["antibiotic", "macrolide"]),
    ("levothyroxine", ["Synthroid", "左甲状腺素"], ["thyroid"]),
    ("sertraline", ["Zoloft", "舍曲林"], ["ssri", "psychiatry"]),
    ("albuterol", ["Ventolin", "沙丁胺醇"], ["asthma", "beta2"]),
]

# 常用 section：你可以继续扩
SECTIONS = [
    ("indications", "info", "indicated_for", "Indications: commonly used for its approved indications."),
    ("dosage", "info", "follow_guidance", "Dosage: follow label/clinician guidance; individualize based on patient factors."),
    ("contraindication", "contraindication", "do_not_use", "Contraindications: do not use in patients with known hypersensitivity to the drug or components."),
    ("warning", "warning", "monitor_or_avoid", "Warnings/Precautions: monitor for clinically significant adverse effects and interactions; use caution in high-risk patients."),
    ("use_in_specific_populations", "caution", "use_with_caution", "Pregnancy/Lactation: limited data; use only if benefit outweighs risk; consult clinician."),
]

POP_MAP = {
    "use_in_specific_populations": ["pregnancy", "lactation", "pediatrics", "geriatrics"],
    "contraindication": [],
    "warning": [],
    "indications": [],
    "dosage": [],
}


def main():
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    docs_path = os.path.join(out_dir, "docs.jsonl")
    chunks_path = os.path.join(out_dir, "chunks.jsonl")

    with open(docs_path, "w", encoding="utf-8") as fd, open(chunks_path, "w", encoding="utf-8") as fc:
        for generic, brands, tags in DRUGS:
            doc_id = f"demo:{generic}:v1"
            title = f"{generic.title()} ({' / '.join(brands)})"

            doc = {
                "doc_id": doc_id,
                "title": title,
                "drug": {"generic": [generic], "brand": brands},
                "source": {
                    "provider": "demo_label",
                    "url": f"local://demo/{generic}",
                    "retrieved_at": TODAY,
                },
                "lang": "en",
                "country": "US",
                "version": "v1",
                "tags": tags,
            }
            fd.write(json.dumps(doc, ensure_ascii=False) + "\n")

            for idx, (sec, risk_level, action, template) in enumerate(SECTIONS, start=1):
                chunk_id = f"{doc_id}#sec={sec}#p={idx}"
                text = template

                # 给不同药加一点“差异感”，避免全是同一句
                if generic in ("enalapril", "losartan", "valsartan") and sec == "use_in_specific_populations":
                    text = "Pregnancy: avoid use when possible due to fetal risk signals for this class; consult clinician. Lactation: consider alternatives."
                if generic in ("warfarin",) and sec == "use_in_specific_populations":
                    text = "Pregnancy: generally avoid due to teratogenicity/bleeding risk; specialist management required. Lactation: consult clinician."
                if generic in ("ibuprofen",) and sec == "warning":
                    text = "Warnings: gastrointestinal bleeding risk, renal risk; use lowest effective dose for shortest duration; caution in elderly."
                if generic in ("aspirin",) and sec == "warning":
                    text = "Warnings: bleeding risk; avoid in children with viral illness due to Reye syndrome concern; consult clinician."
                if generic in ("metformin",) and sec == "warning":
                    text = "Warnings: lactic acidosis risk in renal impairment; assess kidney function before and during therapy."
                if generic in ("azithromycin",) and sec == "warning":
                    text = "Warnings: QT prolongation risk in susceptible patients; review interacting medications."

                populations = POP_MAP.get(sec, [])

                chunk = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "title": f"{generic.title()} - {sec}",
                    "text": text,
                    "metadata": {
                        "section": sec,
                        "risk_level": risk_level,
                        "action": action,
                        "drug": [generic] + brands,
                        "population": populations,
                        "tags": tags,
                        "source_url": f"local://demo/{generic}",
                        "loc": {"anchor": sec, "paragraph": idx},
                    },
                }
                fc.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("Wrote:")
    print(" -", docs_path)
    print(" -", chunks_path)


if __name__ == "__main__":
    main()
