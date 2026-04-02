# document_priorities.py  v3
#
# Exact filenames verified against git pull output + user's PDF list.
# Added Arab_Convention_Cybercrime_2010_FR.pdf.pdf (note the double .pdf
# extension — that's what DeepL outputs when you translate a PDF named X.pdf).

DOCUMENT_PRIORITIES: dict[str, int] = {

    # ── Priority 1: Core domestic cyber law ──────────────────────────────────
    "DZ_FR_Cybercrime Law_2009.pdf": 1,
    "2016_Algeria_fr_Code Penal.pdf": 1,
    "2018_Algeria_fr_Loi n_ 18-07 du 25 Ramadhan 1439 correspondant au 10 juin 2018 relative à la protection des personnes physiques dans le traitement des données à caractère personnel.pdf": 1,
    "Loi n° 18-07 du 25 Ramadhan 1439 correspondant au 10 juin 2018 relative à la protection des personnes physiques dans le traitement des données à caractère personnel.pdf": 1,

    # ── Priority 2: Supporting domestic law + international ───────────────────
    "Law 20-06 Algeria.pdf": 2,
    "2020_Algeria_fr_Décret présidentiel n_ 20-05 du 24 Joumada El Oula 1441 correspondant au 20 janvier 2020 portant mise en place d_un dispositif national de la sécurité des systèmes d_information.pdf": 2,
    "Loi n∞ 15-04 du 11 Rabie Ethani 1436 correspondant au 1er fÈvrier 2015 fixant les rËgles gÈnÈrales relatives ‡ la signature et ‡ la certification Èlectroniques.pdf": 2,
    "Penal Procedure Code 2021 Update.pdf": 2,
    # Arab Convention — French translation (DeepL output has double .pdf extension)
    "Arab_Convention_Cybercrime_2010_FR.pdf.pdf": 2,
    # Keep English version too in case the translation wasn't done yet
    "2010_en_League of Arab States Convention on Combating Information Technology Offences.pdf": 2,
    "Unofficial French Translation of القانون رقم 03-18 مؤرخ في 9 رمضان عام 1424 الموافق ل4 نوفمبر سنة 2003.pdf": 2,

    # ── Priority 3: Background / supplementary ────────────────────────────────
    "Loi organique n° 12-05 du 18 Safar 1433 correspondant au 12 janvier 2012 relative à l'information.pdf": 3,
    "DZ_AR_CopyrightLaw_2003 2.pdf": 3,
    "Unofficial English Translation of DZ_AR_CopyrightLaw_2003 2.pdf": 3,
}

SKIP_FILES: set[str] = {
    "Unofficial English Translation of DZ_AR_CopyrightLaw_2003 2.pdf",
    "2017_en_Project CyberSouth.pdf",
    "2019_Algeria_en_Expressed Views at the OEWG on Developments in the Field of ICTs.pdf",
    "DZ_AR_Penal_Code_Amendment_2009.pdf",
    "2009_A~1.PDF",
}


def get_priority(filename: str) -> int:
    return DOCUMENT_PRIORITIES.get(filename, 2)


def should_skip(filename: str) -> bool:
    return filename in SKIP_FILES
