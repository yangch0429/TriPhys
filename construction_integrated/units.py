"""units.py  —  SI normalisation of extracted web values, so that "570 g" and "20 ounces" or "2.7 g/cm3" and
"2710 kg/m3" are recognised as one value."""
import re

_F = {
    # mass -> kg
    'mg': (1e-6, 'kg'), 'g': (1e-3, 'kg'), 'gm': (1e-3, 'kg'), 'gram': (1e-3, 'kg'), 'grams': (1e-3, 'kg'), 'kg': (1, 'kg'),
    'kilogram': (1, 'kg'), 'kilograms': (1, 'kg'), 'oz': (0.0283495, 'kg'), 'ounce': (0.0283495, 'kg'),
    'ounces': (0.0283495, 'kg'), 'lb': (0.453592, 'kg'), 'lbs': (0.453592, 'kg'), 'pound': (0.453592, 'kg'),
    'pounds': (0.453592, 'kg'),
    # length -> m
    'mm': (1e-3, 'm'), 'cm': (1e-2, 'm'), 'm': (1, 'm'), 'meter': (1, 'm'), 'meters': (1, 'm'), 'metre': (1, 'm'),
    'metres': (1, 'm'), 'km': (1e3, 'm'), 'in': (0.0254, 'm'), 'inch': (0.0254, 'm'), 'inches': (0.0254, 'm'),
    '"': (0.0254, 'm'), 'ft': (0.3048, 'm'), 'feet': (0.3048, 'm'), 'foot': (0.3048, 'm'), 'um': (1e-6, 'm'),
    'μm': (1e-6, 'm'), 'nm': (1e-9, 'm'),
    # density -> kg/m3
    'g/cm3': (1000, 'kg/m3'), 'g/cc': (1000, 'kg/m3'), 'gcm': (1000, 'kg/m3'), 'g/ml': (1000, 'kg/m3'),
    'kg/l': (1000, 'kg/m3'), 'kg/m3': (1, 'kg/m3'), 'g/l': (1, 'kg/m3'), 'kg/dm3': (1000, 'kg/m3'),
    'lb/in3': (27679.9, 'kg/m3'), 'lb/ft3': (16.0185, 'kg/m3'), 'pcf': (16.0185, 'kg/m3'),
    # acceleration / speed
    'm/s2': (1, 'm/s2'), 'ft/s2': (0.3048, 'm/s2'), 'm/s': (1, 'm/s'), 'km/h': (1 / 3.6, 'm/s'), 'kph': (1 / 3.6, 'm/s'),
    'mph': (0.44704, 'm/s'), 'ft/s': (0.3048, 'm/s'),
    # angle -> deg
    '°': (1, 'deg'), 'deg': (1, 'deg'), 'degree': (1, 'deg'), 'degrees': (1, 'deg'), 'rad': (57.29578, 'deg'),
    # pressure -> Pa
    'pa': (1, 'Pa'), 'kpa': (1e3, 'Pa'), 'mpa': (1e6, 'Pa'), 'gpa': (1e9, 'Pa'), 'bar': (1e5, 'Pa'), 'atm': (101325, 'Pa'),
    'psi': (6894.76, 'Pa'), 'n/m2': (1, 'Pa'), 'n/mm2': (1e6, 'Pa'),
    # viscosity -> Pa s
    'pas': (1, 'Pa s'), 'mpas': (1e-3, 'Pa s'), 'cp': (1e-3, 'Pa s'),
    'centipoise': (1e-3, 'Pa s'), 'p': (0.1, 'Pa s'), 'poise': (0.1, 'Pa s'),
    # energy / heat
    'j/kgk': (1, 'J/kgK'), 'j/kg°c': (1, 'J/kgK'), 'j/gk': (1e3, 'J/kgK'), 'j/g°c': (1e3, 'J/kgK'), 'kj/kgk': (1e3, 'J/kgK'),
    'j/kg': (1, 'J/kg'), 'kj/kg': (1e3, 'J/kg'), 'j/g': (1e3, 'J/kg'), 'cal/g': (4184, 'J/kg'),
    'w/mk': (1, 'W/mK'), 'w/m°c': (1, 'W/mK'),
    'n/m': (1, 'N/m'), 'mn/m': (1e-3, 'N/m'), 'dyn/cm': (1e-3, 'N/m'), 'n': (1, 'N'), 'w': (1, 'W'), 'kw': (1e3, 'W'),
    'hz': (1, 'Hz'), 's': (1, 's'), 'ms': (1e-3, 's'),
    '%': (0.01, '1'), 'percent': (0.01, '1'), '': (1, '1'), 'dimensionless': (1, '1'), 'none': (1, '1'),
}


def key(unit):
    u = str(unit or '').strip().lower()
    if u == '"':
        return u
    u = u.replace('³', '3').replace('²', '2').replace('^', '').replace('−', '-').replace(' per ', '/')
    u = re.sub(r'[\s()*]', '', u)
    if re.fullmatch(r'w[x·.]?/?[x·.]?m[x·.]?(k|°c)(-1)?|w[x·.]?m-1[x·.]?k-1', u):
        return 'w/mk'
    u = u.replace('·', '').replace('.', '')
    return {'g/cm': 'g/cm3', 'gcm-3': 'g/cm3', 'gmcm3': 'g/cm3', 'gm/cm3': 'g/cm3', 'g/cm-3': 'g/cm3', 'kgm-3': 'kg/m3',
            'kg/m': 'kg/m3', 'ms-2': 'm/s2', 'fts-2': 'ft/s2', 'gcm': 'g/cm3', 'g/mlat20c': 'g/ml'}.get(u, u)


def si(value, unit):
    """value (number or [lo, hi]) + unit -> ((lo, hi) in SI, dimension) ; unknown unit -> (values, normalised unit string)"""
    v = value if isinstance(value, list) else [value, value]
    try:
        v = [float(x) for x in v]
    except Exception:
        return None
    lo, hi = min(v), max(v)
    k = key(unit)
    f, d = _F.get(k, (1, k))
    return (lo * f, hi * f), d


def agree(a, b, tol=0.05):
    """a, b = (lo, hi) SI intervals of one dimension: points within tol, ranges with ends within 3 %, point inside range"""
    (al, ah), (bl, bh) = a, b
    pa, pb = al == ah, bl == bh
    if pa and pb:
        return abs(al - bl) <= tol * max(abs(bl), abs(al), 1e-12)
    if not pa and not pb:
        return abs(al - bl) <= 0.03 * max(abs(bl), 1e-12) and abs(ah - bh) <= 0.03 * max(abs(bh), 1e-12)
    p, (l, h) = (al, (bl, bh)) if pa else (bl, (al, ah))
    return l - 0.02 * abs(l) <= p <= h + 0.02 * abs(h)


def width(iv):
    lo, hi = iv
    return (hi - lo) / max(abs(hi + lo) / 2, 1e-12)
