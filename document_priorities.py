# document_priorities.py
#
# This is the authoritative mapping of every PDF in knowledge_base/ to its
# retrieval priority tier.
#
# Priority 1 — Primary Algerian law: these are consulted first on every query.
# Priority 2 — Supporting law & international context: used when P1 is insufficient.
# Priority 3 — Background / supplementary: used as a last resort.
#
# When you add a new PDF to knowledge_base/, add it here too.
# The key is the exact filename (case-sensitive). The value is 1, 2, or 3.

DOCUMENT_PRIORITIES: dict[str, int] = {
    # ── Priority 1: Core domestic cyber law ──────────────────────────────────
    "DZ_FR_Cybercrime Law_2009.pdf": 1,
    "Loi n° 18-07 du 25 Ramadhan 1439 correspondant au 10 juin 2018 relative à la protection des personnes physiques dans le traitement des données à caractère personnel.pdf": 1,
    "2016_Algeria_fr_Code Penal.pdf": 1,           # only relevant articles extracted
    "DZ_AR_Penal_Code_Amendment_2009.pdf": 1,      # FR translation expected in folder

    # ── Priority 2: Supporting domestic law & international context ───────────
    "2020_Algeria_fr_Décret présidentiel n_ 20-05 du 24 Joumada El Oula 1441 correspondant au 20 janvier 2020 portant mise en place d_un dispositif national de la sécurité des systèmes d_information.pdf": 2,
    "Law 20-06 Algeria.pdf": 2,
    "Loi n∞ 15-04 du 11 Rabie Ethani 1436 correspondant au 1er fÈvrier 2015 fixant les rËgles gÈnÈrales relatives ‡ la signature et ‡ la certification Èlectroniques.pdf": 2,
    "Penal Procedure Code 2021 Update.pdf": 2,
    "2010_en_League of Arab States Convention on Combating Information Technology Offences.pdf": 2,  # FR translation expected
    "DZ_AR_CopyrightLaw_2003 2.pdf": 2,            # FR translation expected
    "Unofficial French Translation of القانون رقم 03-18 مؤرخ في 9 رمضان عام 1424 الموافق ل4 نوفمبر سنة 2003.pdf": 2,

    # ── Priority 3: Background / supplementary ────────────────────────────────
    "Loi organique n° 12-05 du 18 Safar 1433 correspondant au 12 janvier 2012 relative à l'information.pdf": 3,
    "2017_en_Project CyberSouth.pdf": 3,
    "2019_Algeria_en_Expressed Views at the OEWG on Developments in the Field of ICTs.pdf": 3,
    "Unofficial English Translation of DZ_AR_CopyrightLaw_2003 2.pdf": 3,
    "القانون رقم 03-18 مؤرخ في 9 رمضان عام 1424 الموافق ل4 نوفمبر سنة 2003, يتضمن الموالفقة على الأمر رقم 03-06 المؤرخ في جمادى الأولى عام 1424 الموافق ل19 يوليو سنة 2003 والمتعلق بالعلامات.pdf": 3,

    # ── Course materials (add your teacher's PDFs here at priority 1) ─────────
    # "ALG_Cyber_2009_FR_0.pdf": 1,
    # "Charte_du_citoyen_numérique.pdf": 1,
    # "Loi-N°18-07.pdf": 1,
}


def get_priority(filename: str) -> int:
    """
    Returns the priority tier for a given PDF filename.
    Falls back to priority 2 if the file isn't explicitly listed,
    so nothing is silently excluded.
    """
    return DOCUMENT_PRIORITIES.get(filename, 2)
