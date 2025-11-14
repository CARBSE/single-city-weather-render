# app.py (replaceable - complete)
import os
import json
import time
import logging
from pathlib import Path
from io import BytesIO
from functools import wraps

from flask import Flask, render_template, request, send_file, jsonify, abort
from flask_caching import Cache

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# optional boto3 for S3 syncing
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO3_AVAILABLE = True
except Exception:
    BOTO3_AVAILABLE = False

# --------------------
# App & config
# --------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
cache = Cache(app, config={"CACHE_TYPE": "simple"})
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data_private"
DATA_DIR = Path(os.environ.get("CARBSE_DATA_DIR", str(DEFAULT_DATA_DIR)))

# image/static folders
STATIC_DIR = BASE_DIR / "static"
IMAGE_DIR = STATIC_DIR / "images"
FIRST_STAGE_DIR = IMAGE_DIR / "new-city-images"
CITY_PROFILE_DIR = IMAGE_DIR / "city-profile-for-single-city"

WEATHER_LOC_FILE = DATA_DIR / "WeathertoolLocations.xlsx"
ADAPTIVE_MODELS_FILE = DATA_DIR / "AdaptiveModels.xlsx"

# S3 settings (if you want to keep data private in S3)
S3_BUCKET = os.environ.get("CARBSE_S3_BUCKET", "").strip()
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")

API_KEY = os.environ.get("CARBSE_API_KEY", "")  # optional protection

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
MONTH_MAP = {m: i+1 for i,m in enumerate(MONTHS)}

# --------------------
# S3 helper: sync selected files into DATA_DIR
# --------------------
def s3_sync_data(bucket_name: str, target_dir: Path, prefix: str = ""):
    """Download selected files from S3 into target_dir.
       Requires AWS creds in environment (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).
       Only downloads files we expect: City_Data_*.xls*, WeathertoolLocations.xlsx, AdaptiveModels.xlsx
    """
    if not BOTO3_AVAILABLE:
        app.logger.warning("boto3 not available; cannot sync from S3.")
        return

    if not bucket_name:
        app.logger.info("No S3 bucket configured (CARBSE_S3_BUCKET empty).")
        return

    s3 = boto3.client("s3", region_name=AWS_REGION)
    patterns = ("City_Data_", "WeathertoolLocations.xlsx", "AdaptiveModels.xlsx")
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        objs = []
        for p in pages:
            for obj in p.get("Contents", []):
                key = obj.get("Key")
                objs.append(key)
        if not objs:
            app.logger.info("No objects found in bucket %s with prefix '%s'", bucket_name, prefix)
            return

        for key in objs:
            lower = key.lower()
            if any(pat.lower() in lower for pat in patterns):
                dest = target_dir / Path(key).name
                try:
                    app.logger.info("Downloading s3://%s/%s -> %s", bucket_name, key, dest)
                    s3.download_file(bucket_name, key, str(dest))
                except (ClientError, BotoCoreError) as e:
                    app.logger.exception("Failed to download %s: %s", key, e)
    except Exception:
        app.logger.exception("Error listing/downloading objects from s3://%s", bucket_name)

# --------------------
# Helpers for data & models
# --------------------
def require_api_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if API_KEY:
            key = request.headers.get("X-API-KEY") or request.args.get("api_key")
            if key != API_KEY:
                return jsonify({"error":"unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper

def read_locations():
    if not WEATHER_LOC_FILE.exists():
        app.logger.warning("WeathertoolLocations not found at %s", WEATHER_LOC_FILE)
        return pd.DataFrame()
    try:
        return pd.read_excel(WEATHER_LOC_FILE)
    except Exception:
        app.logger.exception("Error reading WeathertoolLocations.xlsx")
        return pd.DataFrame()

def build_comfort_models_from_data_dir(data_dir: Path):
    data_dir = Path(data_dir)
    files = [p.name for p in data_dir.glob("*.xls*")] if data_dir.exists() else []
    models_map = {}
    am = data_dir / "AdaptiveModels.xlsx"
    if am.exists():
        try:
            df_am = pd.read_excel(am, sheet_name=0, dtype=str)
            for _, r in df_am.iterrows():
                mdl = str(r.get('Model','')).strip()
                fname = str(r.get('DataFilename','') or r.get('Filename','')).strip()
                if mdl and fname:
                    if (data_dir / fname).exists():
                        models_map[mdl] = fname
                    else:
                        candidates = [f for f in files if fname.lower() in f.lower()]
                        if candidates:
                            models_map[mdl] = candidates[0]
            if models_map:
                return models_map
        except Exception:
            app.logger.exception("Failed reading AdaptiveModels.xlsx; falling back to heuristics")

    # heuristics
    for f in files:
        low = f.lower()
        if 'ashrae' in low:
            models_map.setdefault('ASHRAE 55', f)
        if 'imac' in low and ('mm' in low or 'mixed' in low):
            models_map.setdefault('IMAC Mixed Mode', f)
        if 'imac' in low and ('nv' in low or 'natur' in low):
            models_map.setdefault('IMAC Naturally Ventilated', f)
        if 'imac' in low and ('_r' in low or 'res' in low) and 'mixed' not in low:
            models_map.setdefault('IMAC', f)
    if 'ASHRAE 55' not in models_map:
        models_map['ASHRAE 55'] = 'City_Data_ASHRAE55.xlsx'
    return models_map

def get_weather_file(model_label):
    fname = COMFORT_MODELS.get(model_label)
    if not fname:
        candidates = list(DATA_DIR.glob("City_Data_*.xls*")) if DATA_DIR.exists() else []
        if candidates:
            fname = candidates[0].name
        else:
            return None
    p = DATA_DIR / fname
    return p if p.exists() else None

def load_model_sheet(sheet_name, model_label):
    p = get_weather_file(model_label)
    if not p:
        return pd.DataFrame()
    try:
        return pd.read_excel(p, sheet_name=sheet_name)
    except Exception:
        app.logger.exception("Error reading sheet %s from %s", sheet_name, p)
        return pd.DataFrame()

def find_file_case_insensitive(directory: Path, base_name_noext: str, exts=('.jpg','.jpeg','.png','.webp')):
    if not directory:
        return None
    directory = Path(directory)
    if not directory.exists():
        return None
    base_low = base_name_noext.strip().lower()
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if p.stem.lower() == base_low and p.suffix.lower() in exts:
            return p
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if base_low in p.name.lower():
            return p
    return None

# --------------------
# Chart code (unchanged logic)
# --------------------
def generate_chart_band_or_stacked(df, kind='band', title='Comfort Chart', min_y=6, max_y=None):
    df = df.copy()
    if 'Month' not in df.columns and set(MONTHS).issubset(set(df.columns)):
        if 'Particular' in df.columns:
            mean_row = df[df['Particular'].astype(str).str.strip().str.lower().isin(['mean','mean '])].head(1)
            if not mean_row.empty:
                vals = [mean_row.iloc[0].get(m) for m in MONTHS]
                plot_df = pd.DataFrame({'Month': MONTHS, 'Toutdoorm': vals})
                df = plot_df
            else:
                vals = df.iloc[0][MONTHS].values
                plot_df = pd.DataFrame({'Month': MONTHS, 'Toutdoorm': vals})
                df = plot_df
        else:
            vals = df.iloc[0][MONTHS].values
            plot_df = pd.DataFrame({'Month': MONTHS, 'Toutdoorm': vals})
            df = plot_df

    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(9,4))

    try:
        if df['Month'].dtype == object:
            df['Mnum'] = df['Month'].map(MONTH_MAP)
        else:
            df['Mnum'] = df['Month'].astype(int)
    except Exception:
        df['Mnum'] = np.arange(1, len(df)+1)

    if kind == 'band':
        cols_needed = [
            'Too Hot (< 80% Acceptable)',
            'Warm (80-90% Acceptable)',
            'Comfortable (90% Acceptable)',
            'Cold (80-90% Acceptable)',
            'Too Cold (< 80% Acceptable)'
        ]
        present = all(c in df.columns for c in cols_needed)
        if present and 'Toutdoorm' in df.columns:
            ax.fill_between(df['Mnum'], df['Too Hot (< 80% Acceptable)'], df['Warm (80-90% Acceptable)'], alpha=0.5)
            ax.fill_between(df['Mnum'], df['Warm (80-90% Acceptable)'], df['Comfortable (90% Acceptable)'], alpha=0.5)
            ax.fill_between(df['Mnum'], df['Comfortable (90% Acceptable)'], df['Cold (80-90% Acceptable)'], alpha=0.5)
            ax.fill_between(df['Mnum'], df['Cold (80-90% Acceptable)'], df['Too Cold (< 80% Acceptable)'], alpha=0.5)
            ax.fill_between(df['Mnum'], df['Too Cold (< 80% Acceptable)'], min_y, alpha=0.5)
            ax.plot(df['Mnum'], df['Toutdoorm'], 'k--', linewidth=1.2, label='30 Day Running Mean')
            ax.set_ylabel('Operative Temperature (°C)')
        else:
            if 'Toutdoorm' in df.columns:
                ax.plot(df['Mnum'], df['Toutdoorm'], marker='o', linewidth=2)
                ax.set_ylabel('Value')
            else:
                ax.text(0.5, 0.5, "No monthly data", ha='center', va='center')
                ax.axis('off')
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0); plt.close(fig)
                return buf

    elif kind == 'stacked':
        cols = [
            'Too Cold (< 80% Acceptable)',
            'Cold (80-90% Acceptable)',
            'Comfortable (90% Acceptable)',
            'Warm (80-90% Acceptable)',
            'Too Hot (< 80% Acceptable)'
        ]
        if not all(c in df.columns for c in cols):
            ax.text(0.5, 0.5, "No monthly data", ha='center', va='center')
            ax.axis('off')
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0); plt.close(fig)
            return buf
        try:
            df_stack = df.copy()
            totals = df_stack[cols].sum(axis=1)
            totals = totals.replace(0, np.nan)
            df_stack[cols] = df_stack[cols].div(totals, axis=0) * 100
            bottom = np.zeros(len(df_stack))
            for c in cols:
                vals = df_stack[c].values
                ax.bar(df_stack['Mnum'], vals, bottom=bottom, alpha=0.8, label=c)
                bottom += vals
            ax.set_ylabel('% of Hours')
            ax.set_ylim(0, 100)
            ax.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        except Exception:
            ax.text(0.5, 0.5, "No monthly data", ha='center', va='center')
            ax.axis('off')
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0); plt.close(fig)
            return buf

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS)
    ax.set_xlabel('Month')
    ax.grid(linestyle='--', alpha=0.5)
    ax.set_title(title or "")

    if max_y is not None and kind == 'band':
        ax.set_ylim(bottom=min_y, top=max_y)

    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# --------------------
# Map composite (uses provided map_pin icon if present)
# --------------------
BASE_MAP_FILENAMES = ["india_map.jpg", "india_map_preview.jpg", "india_map_preview.jpeg", "india_map.png"]
PIN_ICON_FILENAME = IMAGE_DIR / "map_pin.png"

def composite_map(city):
    base_path = None
    for fname in BASE_MAP_FILENAMES:
        cand = FIRST_STAGE_DIR / fname
        if cand.exists():
            base_path = cand; break
    if base_path is None:
        for fname in BASE_MAP_FILENAMES:
            cand = IMAGE_DIR / fname
            if cand.exists():
                base_path = cand; break

    if base_path:
        try:
            base = Image.open(base_path).convert('RGBA')
            W, H = base.size

            # full overlay preference
            overlay = None
            overlay_path = None
            for d in (FIRST_STAGE_DIR, CITY_PROFILE_DIR, IMAGE_DIR):
                if not d:
                    continue
                p = find_file_case_insensitive(d, city)
                if p:
                    overlay_path = p
                    try:
                        overlay = Image.open(p).convert('RGBA')
                        break
                    except Exception:
                        app.logger.exception("Failed opening overlay %s", p)
                        overlay = None
                        overlay_path = None

            if overlay is not None and overlay.size == base.size:
                try:
                    out = Image.alpha_composite(base, overlay)
                    return out
                except Exception:
                    app.logger.exception("alpha_composite failed for %s", overlay_path)

            loc_df = read_locations()
            if not loc_df.empty and 'City' in loc_df.columns:
                row = loc_df[loc_df['City'].astype(str).str.lower() == city.lower()]
                if not row.empty:
                    try:
                        x = int(row.iloc[0].get('X', 0) or 0)
                        y = int(row.iloc[0].get('Y', 0) or 0)
                    except Exception:
                        x = 0; y = 0
                    x = max(0, min(W-1, x))
                    y = max(0, min(H-1, y))

                    if PIN_ICON_FILENAME.exists():
                        try:
                            pin = Image.open(PIN_ICON_FILENAME).convert('RGBA')
                            target_h = max(16, int(min(W, H) * 0.05))
                            w0, h0 = pin.size
                            scale = target_h / float(h0)
                            tw = int(w0 * scale)
                            th = int(h0 * scale)
                            pin = pin.resize((tw, th), Image.LANCZOS)
                            offset_x = x - tw // 2
                            offset_y = y - int(th * 0.9)
                            offset_x = max(0, min(W - tw, offset_x))
                            offset_y = max(0, min(H - th, offset_y))
                            base.paste(pin, (offset_x, offset_y), pin)
                        except Exception:
                            app.logger.exception("Failed to paste pin icon; falling back to draw")
                            draw = ImageDraw.Draw(base)
                            r = max(6, int(min(W, H) * 0.012))
                            outline = (0,0,0,255)
                            fill = (255,255,255,255)
                            draw.ellipse((x-r-1, y-r-1, x+r+1, y+r+1), fill=outline)
                            draw.ellipse((x-r, y-r, x+r, y+r), fill=fill)
                    else:
                        draw = ImageDraw.Draw(base)
                        r = max(6, int(min(W, H) * 0.012))
                        outline = (0,0,0,255)
                        fill = (255,255,255,255)
                        draw.ellipse((x-r-1, y-r-1, x+r+1, y+r+1), fill=outline)
                        draw.ellipse((x-r, y-r, x+r, y+r), fill=fill)
            return base
        except Exception:
            app.logger.exception("Error composing base map")
    img = Image.new('RGBA', (600, 600), (247,247,247,255))
    d = ImageDraw.Draw(img)
    d.text((300, 60), "India Map (placeholder)", anchor="mm", fill=(60,60,60))
    return img

# --------------------
# City profile endpoint
# --------------------
@app.route('/city_profile')
def city_profile():
    city = (request.args.get('city') or '').strip()
    if not city:
        abort(400)
    candidates = [CITY_PROFILE_DIR, IMAGE_DIR, FIRST_STAGE_DIR]
    found = None
    for d in candidates:
        if not d:
            continue
        p = find_file_case_insensitive(d, city)
        if p:
            found = p
            break
    if found:
        try:
            return send_file(str(found), conditional=True)
        except Exception:
            app.logger.exception("Failed to send city profile %s", found)
    buf = BytesIO()
    placeholder = Image.new('RGB', (300, 180), color=(245,245,245))
    draw = ImageDraw.Draw(placeholder)
    draw.text((150,90), f"{city}\n(profile)", anchor="mm", fill=(80,80,80))
    placeholder.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# --------------------
# Landing / health (Render friendly)
# --------------------
@app.route('/health')
def health():
    return jsonify({"status":"ok"}), 200

@app.route('/landing')
def landing():
    return render_template('landing.html')

# --------------------
# Main endpoints (same logic)
# --------------------
@app.route('/')
def home():
    df = read_locations()
    states = []
    typologies = []
    if not df.empty and 'State' in df.columns:
        states = sorted(df['State'].dropna().unique().tolist())
    typ_to_models = {}
    if ADAPTIVE_MODELS_FILE.exists():
        try:
            am = pd.read_excel(ADAPTIVE_MODELS_FILE, sheet_name=0, dtype=str)
            for _, r in am.iterrows():
                model = str(r.get('Model','')).strip()
                typs = (str(r.get('Typologies','')).strip() or str(r.get('Typology','')).strip())
                if not model:
                    continue
                if typs:
                    for t in [x.strip() for x in typs.split(',') if x.strip()]:
                        typ_to_models.setdefault(t, []).append(model)
                else:
                    typ_to_models.setdefault('All', []).append(model)
            typologies = sorted(typ_to_models.keys())
        except Exception:
            app.logger.exception("Reading AdaptiveModels failed")
            typ_to_models = {}
            typologies = []
    else:
        typologies = ["Residential","Commercial"]
        typ_to_models = { "Residential": list(COMFORT_MODELS.keys()), "Commercial": list(COMFORT_MODELS.keys()) }

    try:
        typ_to_models_json = json.dumps(typ_to_models)
    except Exception:
        typ_to_models_json = json.dumps({k:list(v) for k,v in typ_to_models.items()})

    return render_template('home.html',
                           states=states, typologies=typologies,
                           typ_to_models=typ_to_models_json,
                           ts=int(time.time()))

@app.route('/api/typologies')
def api_typologies():
    if ADAPTIVE_MODELS_FILE.exists():
        try:
            am = pd.read_excel(ADAPTIVE_MODELS_FILE, sheet_name=0, dtype=str)
            mapping = {}
            for _, r in am.iterrows():
                model = str(r.get('Model','')).strip()
                typs = (str(r.get('Typologies','')).strip() or str(r.get('Typology','')).strip())
                if not model:
                    continue
                if typs:
                    for t in [x.strip() for x in typs.split(',') if x.strip()]:
                        mapping.setdefault(t, []).append(model)
                else:
                    mapping.setdefault('All', []).append(model)
            if mapping:
                return jsonify(mapping)
        except Exception:
            app.logger.exception("api_typologies: AdaptiveModels read failed")
    default = {"Residential": list(COMFORT_MODELS.keys()), "Commercial": list(COMFORT_MODELS.keys())}
    return jsonify(default)

@app.route('/api/states')
def api_states():
    df = read_locations()
    if df.empty or 'State' not in df.columns:
        return jsonify({"states":[]})
    return jsonify({"states": sorted(df['State'].dropna().unique().tolist())})

@app.route('/api/cities')
def api_cities():
    state = request.args.get('state','').strip()
    df = read_locations()
    if df.empty or 'State' not in df.columns or 'City' not in df.columns:
        return jsonify({"cities":[]})
    if state:
        cities = sorted(df[df['State'].astype(str).str.strip()==state]['City'].dropna().unique().tolist())
    else:
        cities = sorted(df['City'].dropna().unique().tolist())
    return jsonify({"cities": cities})

@app.route('/api/city_meta')
def api_city_meta():
    df = read_locations()
    if df.empty:
        return jsonify({})
    out = {}
    for _, r in df.iterrows():
        city = str(r.get('City','')).strip()
        if not city:
            continue
        out[city] = {
            "State": str(r.get('State','')).strip(),
            "X": r.get('X', None),
            "Y": r.get('Y', None)
        }
    return jsonify(out)

@app.route('/analyze', methods=['POST'])
def analyze():
    city = (request.form.get('city') or (request.json.get('city') if request.json else None)) or ''
    model = request.form.get('model','ASHRAE 55') or (request.json.get('model') if request.json else 'ASHRAE 55')
    if not city:
        return jsonify({"error":"city required"}), 400
    df = load_model_sheet('MeanMaxMin', model)
    if df.empty or 'City' not in df.columns:
        return jsonify({"table_html":"<div class='alert alert-warning'>No data available.</div>"})
    row = df[df['City'].astype(str).str.lower()==city.lower()]
    if row.empty:
        return jsonify({"table_html":"<div class='alert alert-warning'>City not found.</div>"})
    table_html = row.to_html(index=False, classes='table table-bordered table-sm')
    return jsonify({"table_html": table_html})

@app.route('/chart_comfort')
@cache.cached(timeout=600, query_string=True)
def chart_comfort():
    city = request.args.get('city','').strip()
    model = request.args.get('model','ASHRAE 55')
    candidate_sheets = ['RawData', 'ComfortBand', 'RawDataComfort', 'RawData24', 'MeanMaxMin']
    for sheet in candidate_sheets:
        df = load_model_sheet(sheet, model)
        if df.empty:
            continue
        if 'City' in df.columns:
            df_city = df[df['City'].astype(str).str.lower() == city.lower()]
        else:
            df_city = df
        if df_city.empty:
            continue
        buf = generate_chart_band_or_stacked(df_city, kind='band', title=city, min_y=6, max_y=None)
        return send_file(buf, mimetype='image/png')
    abort(404)

@app.route('/chart_24x7')
@cache.cached(timeout=600, query_string=True)
def chart_24x7():
    city = request.args.get('city','').strip()
    model = request.args.get('model','ASHRAE 55')
    candidate_sheets = ['RawData24', 'RawData_24', 'RawData24_Hours', 'RawData']
    for sheet in candidate_sheets:
        df = load_model_sheet(sheet, model)
        if df.empty:
            continue
        if 'City' in df.columns:
            df_city = df[df['City'].astype(str).str.lower() == city.lower()]
        else:
            df_city = df
        if df_city.empty:
            continue
        buf = generate_chart_band_or_stacked(df_city, kind='stacked', title=f"{city} 24×7")
        return send_file(buf, mimetype='image/png')
    abort(404)

@app.route('/chart_9x6')
@cache.cached(timeout=600, query_string=True)
def chart_9x6():
    city = request.args.get('city','').strip()
    model = request.args.get('model','ASHRAE 55')
    candidate_sheets = ['RawData8', 'RawData_9x6', 'RawData8_9x6', 'RawData']
    for sheet in candidate_sheets:
        df = load_model_sheet(sheet, model)
        if df.empty:
            continue
        if 'City' in df.columns:
            df_city = df[df['City'].astype(str).str.lower() == city.lower()]
        else:
            df_city = df
        if df_city.empty:
            continue
        buf = generate_chart_band_or_stacked(df_city, kind='stacked', title=f"{city} 9×6")
        return send_file(buf, mimetype='image/png')
    abort(404)

@app.route('/map_preview')
def map_preview():
    city = request.args.get('city','').strip()
    img = composite_map(city)
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/download/stats')
def download_stats():
    city = request.args.get('city','').strip()
    model = request.args.get('model','ASHRAE 55')
    df = load_model_sheet('MeanMaxMin', model)
    if df.empty or 'City' not in df.columns:
        abort(404)
    row = df[df['City'].astype(str).str.lower() == city.lower()]
    if row.empty:
        abort(404)
    out = BytesIO()
    row.to_csv(out, index=False)
    out.seek(0)
    return send_file(out, as_attachment=True, download_name=f"{city}_{model}_stats.csv", mimetype='text/csv')

# --------------------
# Startup syncing (local fallback first, then S3)
# --------------------
def ensure_data_available():
    if DATA_DIR.exists() and any(DATA_DIR.glob("City_Data_*.xls*")) and (DATA_DIR / "WeathertoolLocations.xlsx").exists():
        app.logger.info("Using local DATA_DIR at %s", DATA_DIR)
    else:
        app.logger.info("Local data_private missing or incomplete; attempting S3 sync if configured.")
        if S3_BUCKET:
            if not BOTO3_AVAILABLE:
                app.logger.warning("boto3 not installed; cannot download from S3. Install boto3 and set AWS creds.")
            else:
                try:
                    s3_sync_data(S3_BUCKET, DATA_DIR)
                except Exception:
                    app.logger.exception("S3 sync failed")
        else:
            app.logger.warning("No S3 bucket configured (CARBSE_S3_BUCKET). Place your Excel files in data_private/")

# global comfort models mapping
@app.before_first_request
def prepare():
    ensure_data_available()
    global COMFORT_MODELS
    COMFORT_MODELS = build_comfort_models_from_data_dir(DATA_DIR)
    app.logger.info("COMFORT_MODELS mapping: %s", COMFORT_MODELS)

# --------------------
# Run (development)
# --------------------
if __name__ == '__main__':
    if not DATA_DIR.exists():
        app.logger.warning("CARBSE_DATA_DIR does not exist: %s", DATA_DIR)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
