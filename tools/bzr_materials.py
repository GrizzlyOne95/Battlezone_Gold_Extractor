"""Check explicit Ogre imports needed by generated BZR materials.

This checks inheritance/import visibility, not native shader compilation.
"""
import re

IMPORT='import * from "BZBase.material"'


def with_stock_import(text):
    if re.search(r'^\s*import\s+\*\s+from\s+"?BZBase\.material"?\s*$',text,re.M):return text
    return IMPORT+'\n\n'+text


def validate_stock_inheritance(text,stock):
    def clean(value):return re.sub(r'//[^\n]*|/\*.*?\*/','',value,flags=re.S)
    text=clean(text);stock=clean(stock)
    pattern=r'\b(?:abstract\s+)?(?:material|pass)\s+([^\s{}:]+)'
    local=set(re.findall(pattern,text));available=set(re.findall(pattern,stock))
    parents=re.findall(r'\b(?:material\s+[^\s{}:]+|pass(?:\s+[^\s{}:]+)?)\s*:\s*([^\s{}]+)',text)
    external=set(parents)-local
    imported=re.search(r'^\s*import\s+\*\s+from\s+"?BZBase\.material"?\s*$',text,re.M)
    if external:
        if not imported:raise ValueError('Missing explicit BZBase.material import for '+', '.join(sorted(external)))
        missing=external-available
        if missing:raise ValueError('Unresolved material/pass inheritance: '+', '.join(sorted(missing)))
    return dict(explicit_stock_import=bool(imported),stock_parents=sorted(external),local_definitions=len(local),scope='Static import/inheritance visibility; native shader compilation and technique support require runtime qualification')
