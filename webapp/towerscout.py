#
# TowerScout
# A tool for identifying cooling towers from satellite and aerial imagery
#
# TowerScout Team:
# Karen Wong, Gunnar Mein, Thaddeus Segura, Jia Lu
#
# Licensed under CC-BY-NC-SA-4.0
# (see LICENSE.TXT in the root of the repository for details)
#
# Get the directory of the current script (main.py)
import os, sys

current_dir = os.path.dirname(__file__)

# Append the current directory to sys.path
sys.path.append(os.path.abspath(current_dir))

# import basic functionality
import logging
import ts_imgutil
from ts_bmaps import BingMap
from ts_gmaps import GoogleMap
from ts_zipcode import Zipcode_Provider
from ts_events import ExitEvents
import ts_maps
from ts_azmaps import AzureMap
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from ts_azmapmetrics import azTransactions
from authlib.integrations.flask_client import OAuth
from ts_readdetections import SilverTable
from functools import reduce
import asyncio
import jwt
import msal
from datetime import timedelta


from flask import (
    Flask,
    redirect,
    url_for,
    render_template,
    send_from_directory,
    request,
    session,
    Response,
    jsonify,
)
from flask_session import Session
from waitress import serve
import json
from shutil import rmtree
import zipfile
import ssl
import time
import tempfile
from PIL import Image, ImageDraw
import threading
import gc
import datetime
import uuid

dev = 0

MAX_TILES = 1000
MAX_TILES_SESSION = 100000

current_directory = os.getcwd()
model_dir = os.path.join(os.getcwd(), "webapp/model_params/yolov5")

engines = {}

engine_default = None
engine_lock = threading.Lock()

exit_events = ExitEvents()

# Create a dictionary to store session data
sessions = {}
process_status = {}


def find_model(m):
    for engine in engines:
        if m == engines[engine]["file"]:
            return True
    return False


def get_custom_models():
    # TODO: Rework this to load models from Databricks or equivalent
    for f in os.listdir(model_dir):
        if f.endswith(".pt") and not find_model(f):
            add_model(f)


def add_model(m):
    # remove ".pt"
    mid = m[:-3]

    engines[mid] = {
        "id": mid,
        "name": mid,
        "file": m,
        "engine": None,
        "ts": os.path.getmtime(model_dir + m),
    }


# map providers
providers = {
    # 'google': {'id': 'google', 'name': 'Google Maps'},
    "azure": {"id": "azure", "name": "Azure Maps"},
    "bing": {"id": "bing", "name": "Bing Maps"},
}

# other global variables
google_api_key = ""
bing_api_key = ""
azure_api_key = ""
loop = asyncio.get_event_loop()

# prepare uploads directory

uploads_dir = os.path.join(os.getcwd().replace("webapp", ""), "webapp/uploads")
if not os.path.isdir(uploads_dir):
    os.mkdir(uploads_dir)
for f in os.listdir(uploads_dir):
    os.remove(os.path.join(uploads_dir, f))

ssl._create_default_https_context = ssl._create_unverified_context


# variable for zipcode provider
zipcode_lock = threading.Lock()
zipcode_provider = None


# def start_zipcodes():
#     global zipcode_provider
#     with zipcode_lock:
#         print("instantiating zipcode frame, could take 10 seconds ...")
#         zipcode_provider = Zipcode_Provider()


# Flask boilerplate stuff
app = Flask(__name__)


# @app.before_request
# def initialize_msal_client():
#     # Store the MSAL client object in app.config
#     app.config["MSAL_CLIENT"] = _build_msal_app()


# session = Session()
# configure server-side session
SESSION_TYPE = "filesystem"
SESSION_PERMANENT = False
app.config.from_object(__name__)
Session(app)

# route for js code


@app.route("/site/")
def send_site_index():
    return send_site("towerscout.html")


@app.route("/site/<path:path>")
def send_site(path):
    # print("site page requested:",path)
    return send_from_directory("/", path)


@app.route("/about")
def about():
    return render_template("about.html")


# route for images


@app.route("/js/<path:path>")
def send_js(path):
    return send_from_directory("js", path)


# route for images
@app.route("/img/<path:path>")
def send_img(path):
    return send_from_directory("img", path)


# route for custom images
@app.route("/uploads/<path:path>")
def send_upload(path):
    return send_from_directory("uploads", path)


# route for custom images
@app.route("/rm/uploads/<path:path>")
def remove_upload(path):
    os.remove("uploads/" + path)
    print(" upload deleted")
    return "ok"


# route for js code
@app.route("/css/<path:path>")
def send_css(path):
    return send_from_directory("css", path)


CLIENT_ID = ""
CLIENT_SECRET = ""
TENANT_ID = ""
AUTHORITY = "https://login.microsoftonline.com/" + TENANT_ID
REDIRECT_URI = "https://csels-pd-towerscrt-dev-web-01.edav-dev-app.appserviceenvironment.net/towerscoutmainnodetection/getAToken"

SCOPES = ["User.Read"]


# main page route
@app.route("/")
def map_func():
    # Check if the user is authenticated
    # if 'user' not in session:
    #     return redirect(url_for('login'))
    # If the user is authenticated, you can add more validation if needed
    # user_info = session['user']
    # if user_info.get('roles') != 'admin':  # Example of checking user roles
    #     # If user does not have the correct role, redirect to home
    #     return redirect(url_for('home'))

    # # If all validations pass, you can return something (e.g., a success message or data)
    # return None  # No redirection, validation passed
    if dev == 1:
        session["tiles"] = 0

    # init default engine
    # get_engine(None)

    # check for compatible browser
    offset = datetime.timezone(datetime.timedelta(hours=-5))  # Atlanta / CDC
    print("Main page loaded:", datetime.datetime.now(offset), "EST")
    print("Browser:", request.user_agent.string)
    # if not request.user_agent.browser in ['chrome','firefox']:
    #     return render_template('incompatible.html')

    # clean out any temp dirs
    if "tmpdirname" in session:
        rmtree(session["tmpdirname"], ignore_errors=True, onerror=None)
        del session["tmpdirname"]

    # now render the map.html template, inserting the key
    return render_template(
        "towerscout.html",
        bing_map_key=bing_api_key,
        azure_map_key=azure_api_key,
        dev=dev,
    )


# Scopes for accessing the Microsoft Graph API (we don't need to access Graph in your case, but you'll need at least User.Read)


# Initialize the MSAL confidential client
def _build_msal_app():
    return msal.ConfidentialClientApplication(
        CLIENT_ID, authority=AUTHORITY, client_credential=CLIENT_SECRET
    )


# Route to initiate the authentication process (redirects to Microsoft login)
@app.route("/login")
def login():
    msal_app = _build_msal_app()

    # Create the authorization URL to redirect the user for login (if not already logged in)
    auth_url = msal_app.get_authorization_request_url(SCOPES, redirect_uri=REDIRECT_URI)
    return redirect(auth_url)


# Callback route to handle the response from Microsoft after login
@app.route("/towerscoutmainnodetection/getAToken")
def authorized():
    code = request.args.get("code")

    if not code:
        return "Authorization failed", 400

    msal_app = _build_msal_app()

    # Exchange the authorization code for an access token
    result = msal_app.acquire_token_by_authorization_code(
        code, scopes=SCOPES, redirect_uri=REDIRECT_URI
    )

    if "access_token" in result:
        # Store the user's information (including user_id) in the session
        session["user"] = result.get("id_token_claims")
        # return redirect(url_for('/'))
        # now render the map.html template, inserting the key
        return render_template("towerscout.html", bing_map_key=bing_api_key, dev=dev)
    else:
        return "Error: Unable to acquire token", 400


@app.after_request
def add_header(response):
    response.cache_control.max_age = 1
    return response


# Function to decode the JWT token and extract the user ID (oid)
def get_user_id_from_token(token):
    try:
        # Decode the token (without verifying the signature for simplicity)
        # You may want to verify the token using MS Entra ID's public keys for production
        decoded_token = jwt.decode(token, options={"verify_signature": False})

        # Extract the user ID from the decoded token (oid is the claim representing the user ID)
        user_id = decoded_token.get(
            "oid"
        )  # 'oid' is the claim representing the user ID
        return user_id
    except Exception as e:
        return None


# retrieve available map providers


@app.route("/getproviders")
def get_providers():
    print("map providers requested")
    result = json.dumps([{"id": k, "name": v["name"]} for (k, v) in providers.items()])
    print(result)
    return result


# zipcode boundary lookup


@app.route("/getzipcode")
def get_zipcode():
    global zipcode_provider
    zipcode = request.args.get("zipcode")
    print("zipcode requested:", zipcode)
    with zipcode_lock:
        if zipcode_provider is None:
            print("instantiating zipcode frame, could take 10 seconds ...")
            zipcode_provider = Zipcode_Provider()
        print("looking up zipcode ...")
        return zipcode_provider.zipcode_polygon(zipcode)


# abort route


@app.route("/abort", methods=["get"])
def abort():
    print(" aborting", id(session))
    exit_events.signal(id(session))
    return "ok"


# Endpoint for polling results
@app.route("/getobjects/<process_id>", methods=["GET"])
def get_objects_process_status(process_id):
    results = process_status.get(process_id, "Unknown process ID")
    return jsonify(results)


# detection route
@app.route("/getobjects", methods=["POST"])
def get_objects():
    try:
        # Create a response object
        response = Response("Starting long task...", status=200)

        # Set the connection headers to keep it open
        response.headers['Connection'] = 'keep-alive'
        response.headers['Keep-Alive'] = 'timeout=600, max=100'  # Keep alive for 10 minutes, max 100 requests

        print(" session:", id(session))
        # print("session(user_id)",session['user'])
        # id_token = request.headers.get('X-MS-TOKEN-AAD-ID-TOKEN')
        # logging.info("id_token:{id_token}")
        # auth_code = request.args.get("code")
        # print("auth_code:", auth_code)
        # # user_id = get_ms_entra_ID(auth_code)
        # # logging.info(f"user_id:{user_id}")
        # # Get the token from the Authorization header
        # auth_header = request.headers.get("Authorization")
        # print("auth_header", auth_header)
        # if auth_header:
        #     # Extract the token (after 'Bearer ' prefix)
        #     token = auth_header.split(" ")[1]

        #     # Get the user ID from the token
        #     user_id = get_user_id_from_token(token)
        #     session["user_id"] = user_id
        #     if user_id:
        #         print(
        #             jsonify(
        #                 {"message": "Welcome to the Home page!", "user_id": user_id}
        #             )
        #         )
        #     else:
        #         print(jsonify({"error": "Unable to extract user ID"}), 400)
        # else:
        #     print(jsonify({"error": "Authorization header missing"}), 401)
        # check whether this session is over its limit
        if "tiles" not in session:
            session["tiles"] = 0

        print("tiles queried in session:", session["tiles"])
        if session["tiles"] > MAX_TILES_SESSION:
            return "-1"

        # start time, get params
        start = time.time()
        bounds = request.form.get("bounds")
        # engine = request.form.get("engine")
        provider = request.form.get("provider")
        polygons = request.form.get("polygons")
        # id_token = request.headers.get("X-MS-TOKEN-AAD-ID-TOKEN")
        # access_token = request.headers.get("X-MS-TOKEN-AAD-ACCESS-TOKEN")
        # user_id = request.headers.get("X-MS-CLIENT-PRINCIPAL-ID")
        # user_name = request.headers.get("X-MS-CLIENT-PRINCIPAL-NAME")
        # print("id_token:", id_token)
        # print("access_token:", access_token)
        # print("user_id:", user_id)
        # print("user_name:", user_name)
        # print("incoming detection request:")
        # print(" bounds:", bounds)
        # # print(" engine:", engine)
        # print(" map provider:", provider)
        # print(" polygons:", polygons)

        # cropping
        crop_tiles = False

        # make the polygons
        polygons = json.loads(polygons)
        # print(" parsed polygons:", polygons)
        polygons = [ts_imgutil.make_boundary(p) for p in polygons]
        print(" Shapely polygons:", polygons)

        # # get the proper detector
        # det = get_engines(engine)

        # empty results
        results = []

        # create a map provider object
        map = None
        if provider == "bing":
            map = BingMap(bing_api_key)
        elif provider == "google":
            map = GoogleMap(google_api_key)
        elif provider == "azure":
            map = AzureMap(azure_api_key)

        if map is None:
            print(" could not instantiate map provider:", provider)

        # divide the map into 640x640 parts
        tiles, nx, ny, meters, h, w = map.make_tiles(bounds, crop_tiles=crop_tiles)
        timeoutseconds = len(tiles) * 120 # 2 minutes for tile
        response.headers['Keep-Alive'] = f'timeout={timeoutseconds}, max=100'  # Keep alive for 10 minutes, max 100 requests
        print(f" {len(tiles)} tiles, {nx} x {ny}, {meters} x {meters} m")
        # print(" Tile centers:")
        # for c in tiles:
        #   print("  ",c)

        tiles = [t for t in tiles if ts_maps.check_tile_against_bounds(t, bounds)]
        tiles = [t for t in tiles if ts_imgutil.tileIntersectsPolygons(t, polygons)]
        for i, tile in enumerate(tiles):
            tile["id"] = i
        print(" tiles left after viewport and polygon filter:", len(tiles))

        if request.form.get("estimate") == "yes":
            # reset abort flag
            exit_events.alloc(id(session))  # todo: might leak some of these
            print(" returning number of tiles")

            # + ("" if len(tiles) > MAX_TILES else " (exceeds limit)")
            return str(len(tiles))

        if len(tiles) > MAX_TILES:
            print(" ---> request contains too many tiles")
            exit_events.free(id(session))
            return "[]"
        else:
            # tally the new request
            session["tiles"] += len(tiles)

        # main processing:
        # first, clean out the old tempdir
        if "tmpdirname" in session:
            rmtree(session["tmpdirname"], ignore_errors=True, onerror=None)
            print("cleaned up tmp dir", session["tmpdirname"])
            del session["tmpdirname"]

        # Generate a temporary file name (but don't create the file)
        temp_name = tempfile.mktemp()

        # Extract the filename from the full path
        fname = os.path.basename(temp_name)
        
        session["user_id"] = get_current_user()
        user_id = session["user_id"]

        # augment tiles with retrieved filenames
        for i, tile in enumerate(tiles):
            tile['filename'] = user_id+"/"+fname+str(i)+".jpeg"

        meta, request_id = map.get_sat_maps(tiles, loop, fname, user_id)
        session["metadata"] = meta
       
        print(" asynchronously retrieved", len(tiles), "files")

        # check for abort
        if exit_events.query(id(session)):
            print(" client aborted request.")
            exit_events.free(id(session))
            return "[]"

            
        session["tilesinsession"] = tiles #save_tiles_in_session(tiles)
        # Temporary code
        # return tiles

        # #
        # # detect all towers
        # # Sending a request to databricks with a url to the bronze

        # # Need to Add code to read results from EDAV
        stInstance = SilverTable()
        user_id = map.user_id
        request_id = map.request_id

        results_raw = stInstance.get_bboxesfortiles(
            tiles, exit_events, id(session), request_id, user_id
        )
        logging.info("get_bboxesfortiles completed")
        # abort if signaled
        if exit_events.query(id(session)):
            print(" client aborted request.")
            exit_events.free(id(session))
            return "[]"

         # read metadata if present
        for tile in tiles:
            if meta:
                filename = user_id+"/"+fname+str(tile['id'])+".meta.txt"
                with open(filename) as f:
                    tile['metadata'] = map.get_date(f.read())
                    # print(" metadata: "+tile['metadata'])
                    f.close
            else:
                tile['metadata'] = ""
        print("Before make_persistable_tile_results towerscout.py line 553")
        # record some results in session for later saving if desired
        session["detections"] = make_persistable_tile_results(tiles)
        print("After make_persistable_tile_results towerscout.py line 556")
        # # Only for localhost - Azure app services
        # for chunk in results_raw:
        #     if chunk:
        #         print(f"tile results chunk: {chunk}")
        # post-process the results
        results = []
        for result, tile in zip(results_raw, tiles):
            # adjust xyxy normalized results to lat, long pairs
            for i, object in enumerate(result):
                # object['conf'] *= map.checkCutOffs(object) # used to do this before we started cropping
                object["x1"] = tile["lng"] - 0.5 * tile["w"] + object["x1"] * tile["w"]
                object["x2"] = tile["lng"] - 0.5 * tile["w"] + object["x2"] * tile["w"]
                object["y1"] = tile["lat"] + 0.5 * tile["h"] - object["y1"] * tile["h"]
                object["y2"] = tile["lat"] + 0.5 * tile["h"] - object["y2"] * tile["h"]
                object["tile"] = tile["id"]
                object["id_in_tile"] = i
                object["selected"] = object["secondary"] >= 0.35

                # print(" output:",str(object))
            results += result

        # mark results out of bounds or polygon
        for o in results:
            o["inside"] = ts_imgutil.resultIntersectsPolygons(
                o["x1"], o["y1"], o["x2"], o["y2"], polygons
            ) and ts_maps.check_bounds(o["x1"], o["y1"], o["x2"], o["y2"], bounds)
            # print("in " if o['inside'] else "out ", end="")

        # sort the results by lat, long, conf
        results.sort(key=lambda x: x["y1"] * 2 * 180 + 2 * x["x1"] + x["conf"])

        # coaslesce neighboring (in list) towers that are closer than 1 m for x1, y1
        if len(results) > 1:
            i = 0
            while i < len(results) - 1:
                if (
                    ts_maps.get_distance(
                        results[i]["x1"],
                        results[i]["y1"],
                        results[i + 1]["x1"],
                        results[i + 1]["y1"],
                    )
                    < 1
                ):
                    print(" removing 1 duplicate result")
                    results.remove(results[i + 1])
                else:
                    i += 1

        # prepend a pseudo-result for each tile, for debugging
        tile_results = []
        for tile in tiles:
            tile_results.append(
                {
                    "x1": tile["lng"] - 0.5 * tile["w"],
                    "y1": tile["lat"] + 0.5 * tile["h"],
                    "x2": tile["lng"] + 0.5 * tile["w"],
                    "y2": tile["lat"] - 0.5 * tile["h"],
                    "class": 1,
                    "class_name": "tile",
                    "conf": 1,
                    "metadata": tile["metadata"],
                    "url": tile["url"],
                    "selected": True,
                }
            )

        # all done
        selected = str(reduce(lambda a, e: a + (e["selected"]), results, 0))
        print(
            " request complete,"
            + str(len(results))
            + " detections ("
            + selected
            + " selected), elapsed time: ",
            (time.time() - start),
        )
        results = tile_results + results
        # print()

        exit_events.free(id(session))
        print("Before results = json.dumps(results) towerscout.py line 638")
        # with open('large_results_data.json', 'r') as f:
        #     results = json.load(f)  # You could use ijson if the file is too large
        # print("After results = json.dumps(results) towerscout.py line 640")  
        # session["results"] = results
        # # Return the loaded data as JSON response
        # return jsonify(results)
        results = json.dumps(results)
        print("After results = json.dumps(results) towerscout.py line 640")
        session["results"] = results
        return results
    except Exception as e:
        logging.error("Error at %s", "get_objects towerscout.py", exc_info=e)
    except RuntimeError as e:
        logging.error("Error at %s", "get_objects towerscout.py", exc_info=e)
    except SyntaxError as e:
            logging.error("Error at %s", "get_objects ts_maps.py", exc_info=e)

@app.route("/uploadTileImages", methods=["POST"])
def uploadTileImages():
    try:
        print(" session:", id(session))
        # print("session(user_id)",session['user'])
        # id_token = request.headers.get('X-MS-TOKEN-AAD-ID-TOKEN')
        # logging.info("id_token:{id_token}")
        # auth_code = request.args.get("code")
        # print("auth_code:", auth_code)
        # # user_id = get_ms_entra_ID(auth_code)
        # # logging.info(f"user_id:{user_id}")
        # # Get the token from the Authorization header
        # auth_header = request.headers.get("Authorization")
        # print("auth_header", auth_header)
        # if auth_header:
        #     # Extract the token (after 'Bearer ' prefix)
        #     token = auth_header.split(" ")[1]

        #     # Get the user ID from the token
        #     user_id = get_user_id_from_token(token)
        #     session["user_id"] = user_id
        #     if user_id:
        #         print(
        #             jsonify(
        #                 {"message": "Welcome to the Home page!", "user_id": user_id}
        #             )
        #         )
        #     else:
        #         print(jsonify({"error": "Unable to extract user ID"}), 400)
        # else:
        #     print(jsonify({"error": "Authorization header missing"}), 401)
        # check whether this session is over its limit
        if "tiles" not in session:
            session["tiles"] = 0

        print("tiles queried in session:", session["tiles"])
        if session["tiles"] > MAX_TILES_SESSION:
            return "-1"

        
        bounds = request.form.get("bounds")
        # engine = request.form.get("engine")
        provider = request.form.get("provider")
        polygons = request.form.get("polygons")
        # id_token = request.headers.get("X-MS-TOKEN-AAD-ID-TOKEN")
        # access_token = request.headers.get("X-MS-TOKEN-AAD-ACCESS-TOKEN")
        # user_id = request.headers.get("X-MS-CLIENT-PRINCIPAL-ID")
        # user_name = request.headers.get("X-MS-CLIENT-PRINCIPAL-NAME")
        # print("id_token:", id_token)
        # print("access_token:", access_token)
        # print("user_id:", user_id)
        # print("user_name:", user_name)
        # print("incoming detection request:")
        # print(" bounds:", bounds)
        # # print(" engine:", engine)
        # print(" map provider:", provider)
        # print(" polygons:", polygons)

        # cropping
        crop_tiles = False

        # make the polygons
        polygons = json.loads(polygons)
        # print(" parsed polygons:", polygons)
        polygons = [ts_imgutil.make_boundary(p) for p in polygons]
        print(" Shapely polygons:", polygons)

        # # get the proper detector
        # det = get_engines(engine)

        
        # create a map provider object
        map = None
        if provider == "bing":
            map = BingMap(bing_api_key)
        elif provider == "google":
            map = GoogleMap(google_api_key)
        elif provider == "azure":
            map = AzureMap(azure_api_key)

        if map is None:
            print(" could not instantiate map provider:", provider)

        # divide the map into 640x640 parts
        tiles, nx, ny, meters, h, w = map.make_tiles(bounds, crop_tiles=crop_tiles)
        print(f" {len(tiles)} tiles, {nx} x {ny}, {meters} x {meters} m")
        # print(" Tile centers:")
        # for c in tiles:
        #   print("  ",c)

        tiles = [t for t in tiles if ts_maps.check_tile_against_bounds(t, bounds)]
        tiles = [t for t in tiles if ts_imgutil.tileIntersectsPolygons(t, polygons)]
        for i, tile in enumerate(tiles):
            tile["id"] = i
        print(" tiles left after viewport and polygon filter:", len(tiles))

        if request.form.get("estimate") == "yes":
            # reset abort flag
            exit_events.alloc(id(session))  # todo: might leak some of these
            print(" returning number of tiles")

            # + ("" if len(tiles) > MAX_TILES else " (exceeds limit)")
            return str(len(tiles))

        if len(tiles) > MAX_TILES:
            print(" ---> request contains too many tiles")
            exit_events.free(id(session))
            return "[]"
        else:
            # tally the new request
            session["tiles"] += len(tiles)

        # main processing: Uploading Tile images
        # first, clean out the old tempdir
        if "tmpdirname" in session:
            rmtree(session["tmpdirname"], ignore_errors=True, onerror=None)
            print("cleaned up tmp dir", session["tmpdirname"])
            del session["tmpdirname"]

        # Generate a temporary file name (but don't create the file)
        temp_name = tempfile.mktemp()
        
        # Extract the filename from the full path
        fname = os.path.basename(temp_name)

        session["user_id"] = get_current_user()
        user_id = session["user_id"]
        meta, unique_direcotry, user_id = map.get_sat_maps(tiles, loop, fname, user_id)
        logging.info("get_sat_maps completed")
        session['metadata'] = meta
        print(" asynchronously retrieved", len(tiles), "files")
        logging.info("images uploaded....")
        # check for abort
        if exit_events.query(id(session)):
            print(" client aborted request.")
            exit_events.free(id(session))
            return "[]"
        print(" augment tiles with retrieved filenames")
        # augment tiles with retrieved filenames
        for i, tile in enumerate(tiles):
            tile['filename'] = user_id+"/"+fname+str(i)+".jpeg"
        
        session["tilesinsession"] = tiles
        print(" tilesinsession:",str(session["tilesinsession"]))
        print(" tilesinsession:",session["tilesinsession"])
        print(" tiles:",tiles)

        return jsonify({"user_id": user_id, "request_id": unique_direcotry, "tiles_count": len(tiles)})
    except Exception as e:
        logging.error("Error at %s", "uploadTileImages towerscout.py", exc_info=e)
    except RuntimeError as e:
        logging.error("Error at %s", "uploadTileImages towerscout.py", exc_info=e)
    except SyntaxError as e:
        logging.error("Error at %s", "uploadTileImages ts_maps.py", exc_info=e)

@app.route("/pollSilverTable", methods=["POST"])
def pollSilverTable():
    try:
        
        print(" session:", id(session))
        
        # # Need to Add code to read results from EDAV
        stInstance = SilverTable()
        user_id = request.form.get("user_id")
        request_id = request.form.get("request_id")
        tilescount = int(request.form.get("tiles_count"))
        jobDone = stInstance.poll_SilverTableJobDone(request_id, user_id, tilescount, 10)
        logging.info("pollSilverTable completed")
        # abort if signaled
        if exit_events.query(id(session)):
            print(" client aborted request.")
            exit_events.free(id(session))
            return "[]"
         
        return jsonify({"jobDone": jobDone})
    except Exception as e:
        logging.error("Error at %s", "get_objects towerscout.py", exc_info=e)
    except RuntimeError as e:
        logging.error("Error at %s", "get_objects towerscout.py", exc_info=e)
    except SyntaxError as e:
            logging.error("Error at %s", "get_objects ts_maps.py", exc_info=e)

@app.route("/fetchBoundingBoxResults", methods=["POST"])
def fetchBoundingBoxResults():
    try:
        print(" session:", id(session))
        
        tiles = []
        start = time.time()
        user_id = request.form.get("user_id")
        request_id = request.form.get("request_id")
        bounds = request.form.get("bounds")
        polygons = request.form.get("polygons")
        stInstance = SilverTable()

        # cropping
        crop_tiles = False

        # make the polygons
        polygons = json.loads(polygons)
        # print(" parsed polygons:", polygons)
        polygons = [ts_imgutil.make_boundary(p) for p in polygons]
        print(" Shapely polygons:", polygons)
        
        provider = request.form.get("provider")
        # empty results
        results = []

        # create a map provider object
        map = None
        if provider == "bing":
            map = BingMap(bing_api_key)
        elif provider == "google":
            map = GoogleMap(google_api_key)
        elif provider == "azure":
            map = AzureMap(azure_api_key)

        if map is None:
            print(" could not instantiate map provider:", provider)

         # divide the map into 640x640 parts
        tiles, nx, ny, meters, h, w = map.make_tiles(bounds, crop_tiles=crop_tiles)
        print(f" {len(tiles)} tiles, {nx} x {ny}, {meters} x {meters} m")
        # print(" Tile centers:")
        # for c in tiles:
        #   print("  ",c)
       
        tiles = [t for t in tiles if ts_maps.check_tile_against_bounds(t, bounds)]
        tiles = [t for t in tiles if ts_imgutil.tileIntersectsPolygons(t, polygons)]
        for i, tile in enumerate(tiles):
            tile["id"] = i
            
        results_raw = stInstance.get_bboxesfortilesWithoutPolling(
            tiles, exit_events, id(session), request_id, user_id
        )
        
        logging.info("get_bboxesfortiles completed")
        # abort if signaled
        if exit_events.query(id(session)):
            print(" client aborted request.")
            exit_events.free(id(session))
            return "[]"

         # read metadata if present
        for tile in tiles:
            # if meta:
            #     filename = user_id+"/"+fname+str(tile['id'])+".meta.txt"
            #     with open(filename) as f:
            #         tile['metadata'] = map.get_date(f.read())
            #         # print(" metadata: "+tile['metadata'])
            #         f.close
            # else:
            tile['metadata'] = ""
        print("Before make_persistable_tile_results towerscout.py line 553")
        # record some results in session for later saving if desired
        session["detections"] = make_persistable_tile_results(tiles)
        print("After make_persistable_tile_results towerscout.py line 556")
        # # Only for localhost - Azure app services
        # for chunk in results_raw:
        #     if chunk:
        #         print(f"tile results chunk: {chunk}")
        # post-process the results
               
        results = []
        for result, tile in zip(results_raw, tiles):
            # print(f"tile['lng']: {tile['lng']}")
            # print(f"tile['w']: {tile['w']}")
            # print(f"tile['lat']: {tile['lat']}")
            # print(f"tile['h']: {tile['h']}")
            # adjust xyxy normalized results to lat, long pairs
            for i, object in enumerate(result):
                # object['conf'] *= map.checkCutOffs(object) # used to do this before we started cropping
                object["x1"] = tile["lng"] - 0.5 * tile["w"] + object["x1"] * tile["w"]
                object["x2"] = tile["lng"] - 0.5 * tile["w"] + object["x2"] * tile["w"]
                object["y1"] = tile["lat"] + 0.5 * tile["h"] - object["y1"] * tile["h"]
                object["y2"] = tile["lat"] + 0.5 * tile["h"] - object["y2"] * tile["h"]
                object["tile"] = tile["id"]
                object["id_in_tile"] = i
                object["selected"] = object["secondary"] >= 0.35
                # print(f"object['x1']: {object['x1']}")
                # print(f"object['y1']: {object['y1']}")
                # print(f"object['y2']: {object['y2']}")
                # print(f"object['x1']: {object['x1']}")
                # # print(" output:",str(object))
            results += result

        # mark results out of bounds or polygon
        for o in results:
            o["inside"] = ts_imgutil.resultIntersectsPolygons(
                o["x1"], o["y1"], o["x2"], o["y2"], polygons
            ) and ts_maps.check_bounds(o["x1"], o["y1"], o["x2"], o["y2"], bounds)
            # print("in " if o['inside'] else "out ", end="")

        # sort the results by lat, long, conf
        results.sort(key=lambda x: x["y1"] * 2 * 180 + 2 * x["x1"] + x["conf"])

        # coaslesce neighboring (in list) towers that are closer than 1 m for x1, y1
        if len(results) > 1:
            i = 0
            while i < len(results) - 1:
                if (
                    ts_maps.get_distance(
                        results[i]["x1"],
                        results[i]["y1"],
                        results[i + 1]["x1"],
                        results[i + 1]["y1"],
                    )
                    < 1
                ):
                    print(" removing 1 duplicate result")
                    results.remove(results[i + 1])
                else:
                    i += 1

        # prepend a pseudo-result for each tile, for debugging
        tile_results = []
        for tile in tiles:
            tile_results.append(
                {
                    "x1": tile["lng"] - 0.5 * tile["w"],
                    "y1": tile["lat"] + 0.5 * tile["h"],
                    "x2": tile["lng"] + 0.5 * tile["w"],
                    "y2": tile["lat"] - 0.5 * tile["h"],
                    "class": 1,
                    "class_name": "tile",
                    "conf": 1,
                    "metadata": tile["metadata"],
                    "url": tile["url"],
                    "selected": True,
                }
            )

        # all done
        selected = str(reduce(lambda a, e: a + (e["selected"]), results, 0))
        print(
            " request complete,"
            + str(len(results))
            + " detections ("
            + selected
            + " selected), elapsed time: ",
            (time.time() - start),
        )
        results = tile_results + results
        # print()

        exit_events.free(id(session))
        print("Before results = json.dumps(results) towerscout.py line 638")
        # with open('large_results_data.json', 'r') as f:
        #     results = json.load(f)  # You could use ijson if the file is too large
        # print("After results = json.dumps(results) towerscout.py line 640")  
        # session["results"] = results
        # # Return the loaded data as JSON response
        exit_events.free(id(session))
        # return jsonify(results)
        results = json.dumps(results)
        print("After results = json.dumps(results) towerscout.py line 640")
        session["results"] = results
        return results
    except Exception as e:
        logging.error("Error at %s", "get_objects towerscout.py", exc_info=e)
    except RuntimeError as e:
        logging.error("Error at %s", "get_objects towerscout.py", exc_info=e)
    except SyntaxError as e:
            logging.error("Error at %s", "get_objects ts_maps.py", exc_info=e)

def get_current_user():
   #Implementing for azure app services
    user_id = request.headers.get('X-MS-CLIENT-PRINCIPAL-ID')
    return f"{user_id}"

def get_file_name(file_path):
    file_name_and_extension = os.path.basename(file_path)
    return file_name_and_extension


def allowed_extension(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "png",
        "jpg",
        "jpeg",
    }


# detection route for provided images


@app.route("/getobjectscustom", methods=["POST"])
def get_objects_custom():
    start = time.time()
    # engine = request.form.get("engine")
    print("incoming custom image detection request:")
    # print(" engine:", engine)

    # upload the file
    if request.method != "POST":
        print(" --- POST requests only please")
        return None

    # check if the post request has the file part
    if "image" not in request.files:
        print(" --- no file part in request")
        return None

    file = request.files["image"]
    if file.filename == "":
        print(" --- no selected image file")
        return None

    if not file or not allowed_extension(file.filename):
        print(" --- invalid file or extension:", file.filename)
        return None

    # empty results
    results = []
    # TODO: Setup up webhook or polling to get the results of the cooling tower detection
    session["results"] = results
    return results


#
# pillow helper function to draw
#
def drawResult(r, im):
    print(" drawing ...")
    draw = ImageDraw.Draw(im)
    draw.rectangle(
        [
            im.size[0] * r["x1"],
            im.size[1] * r["y1"],
            im.size[0] * r["x2"],
            im.size[1] * r["y2"],
        ],
        outline="red",
    )


@app.route("/getazmaptransactions", methods=["GET"])
def getazmaptransactions():
    try:

        result = azTransactions.getAZTransactionCount(2)

        return result
    except Exception as e:
        logging.error("Error at %s", "getazmaptransactions towerscout.py", exc_info=e)
    except RuntimeError as e:
        logging.error("Error at %s", "getazmaptransactions towerscout.py", exc_info=e)


# download results as dataset for formal training /testing
@app.route("/getdataset", methods=["POST"])
def send_dataset():
    print("Dataset requested")

    include = json.loads(request.form.get("include"))
    additions = json.loads(request.form.get("additions"))
    tiles = session["detections"]
    meta = session["metadata"]

    # print(" raw inclusions:", request.form.get("include"))
    # print(" inclusions:", include)
    # print(" last result:", json.dumps(tiles))

    # filter to keep only "included" (i.e. selected and meeting threshold) detections
    keep_detections = set([])
    keep_detection_ids = set([])
    for inclusion in include:
        try:
            keep_detections.add(
                tiles[inclusion["tile"]]["detections"][inclusion["detection"]]
            )
            # if the absolute tile id was also included,
            if "id" in inclusion:
                keep_detection_ids.add(inclusion["id"])
        except:
            print(" invalid inclusion:", inclusion)
    print(" writing labels ...")

    # write files and records which ones had detections
    filenames = []
    for i, tile in enumerate(tiles):
        filenames += write_labels(i, tile, keep_detections, additions, not meta)
    # make a new tempdir name and attach to session
    tmpdir = tempfile.TemporaryDirectory()
    tmpdirname = tmpdir.name
    session["tmpdirname"] = tmpdirname
    tmpdir.cleanup() 
    os.mkdir(tmpdirname)
    logging.info("created tmp dir:{tmpdirname}")

    # write a contents file so we can load this again some time
    write_contents_file(
        session["tmpdirname"], tiles, keep_detection_ids, additions, meta
    )

    print(" zipping data ...")
    zipdir(session["tmpdirname"], filenames)
    print(" done.")
    # print()
    return send_from_directory("temp", "dataset.zip")


# adapted from
# https://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory-in-python


def zipdir(path, filenames):
    zipf = zipfile.ZipFile("temp/dataset.zip", "w", zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(path):
        for file in files:
            print(" compressing file", file)
            # select the right folder
            if file.endswith("contents.txt"):
                folder = "."
            elif file.endswith(".meta.txt"):
                continue
            elif file.endswith(".zip"):
                continue
            elif os.path.join(root, file) in filenames:
                if file.endswith(".jpg"):
                    folder = "train/images"
                elif file.endswith(".xml"):
                    folder = "train/labels-xml"
                else:
                    folder = "train/labels"
            else:
                folder = "empty"
            zipf.write(
                os.path.join(root, file),
                os.path.relpath(
                    os.path.join(root, folder, file), os.path.join(path, "..")
                ),
            )
    zipf.close()


# get a portion of the tiles in serializable form to attach to the session
def make_persistable_tile_results(tiles):
    return [
        {
            "filename": tile["filename"],
            "labelfilename": tile["filename"][0:-4] + ".txt",
            "detections": tile["detections"],
            "metadata": tile["metadata"],
            "url": tile["url"],
            "index": tile["id"],
        }
        for tile in tiles
    ]

# get a portion of the tiles in serializable form to attach to the session
def save_tiles_in_session(tiles):

    return [
        {
            # "filename": tile["filename"],
            # "labelfilename": tile["filename"][0:-4] + ".txt",
            "metadata": tile["metadata"],
            "url": tile["url"],
            "index": tile["id"],
        }
        for tile in tiles
    ]

# write out the detections as label files (to download the dataset)
# returns the names of img and label if detections were present, otherwise empty list
def write_labels(tile_id, tile, keep, additions, double_res):
    empty = True
    # print(" attempting to write file ", tile['labelfilename'], "...")

    with open(tile["labelfilename"], "w") as f:
        print(" writing file ", f.name, "...")
        for detection in tile["detections"]:
            if detection in keep:
                # print("", detection)
                f.write(detection)
                empty = False
        for a in additions:
            if a["tile"] == tile_id:
                f.write(
                    " ".join(
                        [
                            "0",
                            str(a["centerx"]),
                            str(a["centery"]),
                            str(a["w"]),
                            str(a["h"]),
                        ]
                    )
                    + "\n"
                )
                empty = False

    # write xml labels for augmentation pipeline
    name = tile["labelfilename"][:-3] + "xml"
    size = 1280 if double_res else 640

    with open(name, "w") as f:
        # print(" writing file ", f.name, "...")
        xml = "<annotation>\n"
        xml += "<size>\n"
        xml += "  <width>" + str(size) + "</width>\n"
        xml += "  <height>" + str(size) + "</height>\n"
        xml += "  <depth>3</depth>\n"
        xml += "</size>\n"
        f.write(xml)
        for detection in tile["detections"]:
            if detection in keep:
                # print("", detection)
                xml = xml_from_label(detection, size)
                f.write(xml)
                empty = False
        for a in additions:
            if a["tile"] == tile_id:
                label = " ".join(
                    [
                        "0",
                        str(a["centerx"]),
                        str(a["centery"]),
                        str(a["w"]),
                        str(a["h"]),
                    ]
                )
                xml = xml_from_label(label, size)
                f.write(xml)
                empty = False
        xml = "</annotation>\n"
        f.write(xml)

    return (
        []
        if empty
        else [
            tile["filename"],
            tile["labelfilename"],
            tile["labelfilename"][:-3] + "xml",
        ]
    )


# convert YOLOv5-style object label to roboflow XML:
def xml_from_label(label, size):
    x = [float(x) for x in label.split(" ")]
    xmin = int((x[1] - 0.5 * x[3]) * size)
    xmax = int((x[1] + 0.5 * x[3]) * size)
    ymin = int((x[2] - 0.5 * x[4]) * size)
    ymax = int((x[2] + 0.5 * x[4]) * size)

    xml = "<object>\n"
    xml += "  <bndbox>\n"
    xml += "    <xmin>" + str(xmin) + "</xmin>\n"
    xml += "    <xmax>" + str(xmax) + "</xmax>\n"
    xml += "    <ymin>" + str(ymin) + "</ymin>\n"
    xml += "    <ymax>" + str(ymax) + "</ymax>\n"
    xml += "  </bndbox>\n"
    xml += "</object>\n"

    return xml


def write_contents_file(tmpdirname, tiles, keep_ids, additions, meta):
    with open(tmpdirname + "/contents.txt", "w") as f:
        f.write("[")
        f.write(json.dumps(tiles))
        f.write(",")
        # if we got information about which ids were still selected, filter the result before recording
        if len(keep_ids) > 0:
            print(" filtering results for", len(keep_ids), "selections")
            results = json.loads(session["results"])
            tile_count = 0

            # first, write write current results that are checked (i.e. in keep_ids)
            for i, result in enumerate(results):
                if result["class_name"] != "tile":
                    result["selected"] = i - tile_count in keep_ids
                    print(
                        "",
                        i - tile_count,
                        "included" if (i - tile_count) in keep_ids else "not included",
                    )
                else:
                    tile_count += 1

            # store the filtered results back in session
            session["results"] = json.dumps(results)

        # now, process additions, and add them to the session results
        # todo!!! Fix the restore bug. Note: Send more lat/long, id_in_tile info from client
        print("Session results before additions: ", session["results"])
        print("JSON version of additions:", json.dumps(additions))
        if len(additions) > 0:
            # make every this:
            #
            pass  # todo!!!

        # write the whole "current result set", modified selections, additions and all
        f.write(session["results"])
        f.write("," + ("false" if meta else "true"))
        f.write("]")



def adapt_filenames(filenames, old_stem, new_stem):
    # print("f[0]", filenames[0])
    # print("old_dir",old_dir)
    results = []
    for f in filenames:
        f = f[len(old_stem) + 1 :]  # strip old dir name
        # print("stripped:",f)
        if f.startswith("empty/"):
            f = f[len("empty/") :]
        elif f.startswith("train/images/"):
            f = f[len("train/images/") :]
        elif f.startswith("train/labels/"):
            f = f[len("train/labels/") :]

        # print("check: f:", f, "old:", old_dir, "new:", new_dir)
        if f.startswith(old_stem):
            f = new_stem + f[len(old_stem) :]

        results.append(f)

    return results


def adapt_tiles(tiles, tmpdirname, old_stem, new_stem):
    for t in tiles:
        # print("before", t['filename'], t['labelfilename'])
        name = t["filename"][t["filename"].rindex("/") + 1 :]
        t["filename"] = tmpdirname + "/" + new_stem + name[len(old_stem) :]
        t["labelfilename"] = (
            tmpdirname + "/" + new_stem + name[len(old_stem) :][0:-4] + ".txt"
        )
        # print("after", t['filename'], t['labelfilename'])
    return tiles


if __name__ == "__main__":
    # read maps api key (not in source code due to security reasons)
    # has to be an api key with access to maps, staticmaps and places
    # todo: deploy CDC-owned key in final version
    with open("apikey.txt") as f:
        azure_api_key = f.readline().split()[0]
        bing_api_key = f.readline().split()[0]
        f.close
    app.config['DEBUG'] = True
    app.config['timeout'] = 3600
    # logging.basicConfig(level=logging.DEBUG)
    # app.logger.setLevel(logging.DEBUG)
    app.permanent_session_lifetime = timedelta(minutes=60)  # Adjust this as needed
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Disable pretty-printing for large JSON



    # # This is for localhost only
    # print("Tower Scout ready on port 5000...")
    # serve(app, host="0.0.0.0", port=5000)
    
    # The following is for Azure app services
    # Azure provides the port via the environment variable 'PORT'
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if not set (useful for local testing)
    app.run(host='0.0.0.0', port=port)  # Listen on all IP addresses
