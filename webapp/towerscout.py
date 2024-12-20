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
# configure server-sise session
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


# def get_ms_entra_ID():

# #     # user_id = request.headers.get('X-MS-CLIENT-PRINCIPAL-ID')
# #     # return user_id
# #     client_id = "6095d61a-3d9c-4499-a4c1-262709bc2044"
# #     client_secret = "5fd05eaa-4b75-4ec1-96d5-8efb87e017b3"
# #     tenant_id = "9ce70869-60db-44fd-abe8-d2767077fc8f"
# #     redirect_url="https://intranet.cdc.gov"

# #     oauth = OAuth(app)
# #     oauth.register(
# #     name='azure',
# #     client_id=client_id,
# #     client_secret=client_secret,
# #     authorize_url=f'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize',
# #     authorize_params=None,
# #     access_token_url=f'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token',
# #     refresh_token_url=None,
# #     client_kwargs={'scope': 'openid profile email'},
# # )
# #     token = oauth.azure.authorize_access_token()

#     # # Decode the ID Token to extract the user ID (sub claim)
#     # user_info = oauth.azure.parse_id_token(token)
#     # user_id = user_info['sub']  # User ID in Azure AD (subject claim)
#     # return user_id
#     # return jsonify(user_info)
#     # Create a ConfidentialClientApplication instance
#     app.config['MSAL_CLIENT'] = ConfidentialClientApplication(
#         client_id=CLIENT_ID,
#         client_credential=CLIENT_SECRET,
#         authority='https://login.microsoftonline.com/9ce70869-60db-44fd-abe8-d2767077fc8f'
#     )
#     auth_code = request.args.get('authorization')

#      # Use the authorization code to acquire an access token and ID token
#     result = app.config['MSAL_CLIENT'].acquire_token_by_authorization_code(
#          auth_code,
#          scopes=['User.Read'],
#          redirect_uri='https://intranet.cdc.gov'
#      )

#     resultstring =json.dumps(result)
#     app.logger.info("result{resultstring}")
#     # logging.info("result{resultstring}")
#     print(json.dumps(resultstring))
#      # Extract and return the user ID from the ID token claims
#     user_id = result.get("id_token_claims", {}).get("sub")
#     return user_id


# detection route
@app.route("/getobjects", methods=["POST"])
def get_objects():
    try:
        print(" session:", id(session))
        # print("session(user_id)",session['user'])
        # id_token = request.headers.get('X-MS-TOKEN-AAD-ID-TOKEN')
        # logging.info("id_token:{id_token}")
        auth_code = request.args.get("code")
        print("auth_code:", auth_code)
        # user_id = get_ms_entra_ID(auth_code)
        # logging.info(f"user_id:{user_id}")
        # Get the token from the Authorization header
        auth_header = request.headers.get("Authorization")
        print("auth_header", auth_header)
        if auth_header:
            # Extract the token (after 'Bearer ' prefix)
            token = auth_header.split(" ")[1]

            # Get the user ID from the token
            user_id = get_user_id_from_token(token)
            session["user_id"] = user_id
            if user_id:
                print(
                    jsonify(
                        {"message": "Welcome to the Home page!", "user_id": user_id}
                    )
                )
            else:
                print(jsonify({"error": "Unable to extract user ID"}), 400)
        else:
            print(jsonify({"error": "Authorization header missing"}), 401)
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
        id_token = request.headers.get("X-MS-TOKEN-AAD-ID-TOKEN")
        access_token = request.headers.get("X-MS-TOKEN-AAD-ACCESS-TOKEN")
        user_id = request.headers.get("X-MS-CLIENT-PRINCIPAL-ID")
        user_name = request.headers.get("X-MS-CLIENT-PRINCIPAL-NAME")
        print("id_token:", id_token)
        print("access_token:", access_token)
        print("user_id:", user_id)
        print("user_name:", user_name)
        print("incoming detection request:")
        print(" bounds:", bounds)
        # print(" engine:", engine)
        print(" map provider:", provider)
        print(" polygons:", polygons)

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

        # make a new tempdir name and attach to session
        tmpdir = tempfile.TemporaryDirectory()
        tmpdirname = tmpdir.name
        # tmpfilename = tmpdirname[tmpdirname.rindex("/")+1:]
        tmpfilename = get_file_name(tmpdirname)
        print("creating tmp dir", tmpdirname)
        session["tmpdirname"] = tmpdirname
        tmpdir.cleanup()  # yeah this is asinine but I need the tmpdir to survive to I will create it manually next
        os.mkdir(tmpdirname)
        print("created tmp dir", tmpdirname)

        # Images get uploaded to datalake witha unique directory name
        # databricks feature - autoloader - writes detections with labels to Silver
        # retrieve tiles and metadata if available
        # user_id = get_ms_entra_ID()
        # print({user_id})
        meta = map.get_sat_maps(tiles, loop, tmpdirname, tmpfilename)
        session["metadata"] = meta
        print(" asynchronously retrieved", len(tiles), "files")

        # check for abort
        if exit_events.query(id(session)):
            print(" client aborted request.")
            exit_events.free(id(session))
            return "[]"

        # augment tiles with retrieved filenames
        for i, tile in enumerate(tiles):
            tile["filename"] = tmpdirname + "/" + tmpfilename + str(i) + ".jpeg"
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
        # abort if signaled
        if exit_events.query(id(session)):
            print(" client aborted request.")
            exit_events.free(id(session))
            return "[]"

        # read metadata if present
        for tile in tiles:
            if meta:
                filename = (
                    tmpdirname + "/" + tmpfilename + str(tile["id"]) + ".meta.txt"
                )
                with open(filename) as f:
                    tile["metadata"] = map.get_date(f.read())
                    # print(" metadata: "+tile['metadata'])
                    f.close
            else:
                tile["metadata"] = ""

        # record some results in session for later saving if desired
        session["detections"] = make_persistable_tile_results(tiles)
        print(f"tile results: {results_raw}")
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
        # Read and return the data (just to show the file is written correctly)
        results = json.dumps(results)
        session["results"] = results
        return results
    except Exception as e:
        logging.error("Error at %s", "division", exc_info=e)


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
        logging.error(e)
    except RuntimeError as e:
        logging.error(e)


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
        print(" writing file ", f.name, "...")
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
            # {"tile": 0, "centerx": 0.9099609375, "centery": 0.6593624174477392, "w": 0.034375, "h": 0.037459364023982526}
            #
            # into this:
            #
            # {"x1": -74.00627583990182, "y1": 40.71050060528311, "x2": -74.0062392291362, "y2": 40.71042470437244,
            #  "conf": 1.0, "class": 0, "class_name": "ct", "secondary": 1..0,
            #  "tile": 1, "id_in_tile": 11, "selected": true, "inside": true}

            pass  # todo!!!

        # write the whole "current result set", modified selections, additions and all
        f.write(session["results"])
        f.write("," + ("false" if meta else "true"))
        f.write("]")

app.permanent_session_lifetime = timedelta(minutes=60)  # Adjust this as needed
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Disable pretty-printing for large JSON
#
#
# upload dataset for further editing:
#

# @app.route('/uploaddataset', methods=['POST'])
# def upload_dataset():
#     print("Dataset upload")

#     # make a temp dir as usual
#     # first, clean out the old tempdir
#     if "tmpdirname" in session:
#         rmtree(session['tmpdirname'], ignore_errors=True, onerror=None)
#         print(" cleaned up tmp dir", session['tmpdirname'])
#         del session['tmpdirname']

#     # make a new tempdir name and attach to session
#     tmpdirname = tempfile.mkdtemp()
#     print(" creating tmp dir", tmpdirname)
#     session['tmpdirname'] = tmpdirname

#     # check if the post request has the file part
#     if 'dataset' not in request.files:
#         print(" --- no file part in request")
#         return None

#     file = request.files['dataset']
#     if file.filename == '':
#         print(' --- no selected dataset file')
#         return None

#     if not file or not file.filename.endswith(".zip"):
#         print(" --- invalid file or extension:", file.filename)
#         return None

#     filename = tmpdirname + "/" + file.filename
#     file.save(filename)
#     new_stem = tmpdirname[tmpdirname.rindex("/")+1:]

#     # unzip dataset.zip
#     # - "empty" tiles and labels right into "."
#     # - "train" combine "images" and "labels" folders into "."
#     # content.txt in "."
#     with zipfile.ZipFile(filename) as zipf:
#         # read previous results and tiles from content.txt and add to session
#         # print(" zip contents:")
#         filenames = zipf.namelist()
#         old_stem = filenames[0][:filenames[0].index("/")]
#         files = adapt_filenames(filenames, old_stem, new_stem)
#         # print(files)
#         for f_zip, f_new in zip(zipf.namelist(), files):
#             print(" processing",f_zip,"to:",f_new)
#             if not f_zip.endswith(".xml"):
#                 with zipf.open(f_zip) as f:
#                     with open(tmpdirname+"/"+f_new, "wb") as f_target:
#                         print(" writing", tmpdirname+"/"+f_new)
#                         f_target.write(f.read())

#     # process contents file
#     results = []
#     print("parsing contents.txt in", tmpdirname)
#     with open(tmpdirname+"/contents.txt") as f:
#         results = json.loads(f.read())

#     session['detections'] = adapt_tiles(
#         results[0], tmpdirname, old_stem, new_stem)
#     session['results'] = json.dumps(results[1])
#     session['metadata'] = results[2]
#     # print("Results:", results[1])
#     # return previous results
#     print(" dataset restored.")

#     return session['results']

# carefully unravel the zip structure we created in the dataset, and make it all flat


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
    # app.run(debug = True)
    # app.secret_key = 'super secret key'
    # app.config['SESSION_TYPE'] = 'filesystem'
    # get_custom_models()
    # engine_default = sorted(engines.items(), key=lambda x: -x[1]["ts"])[0][0]

    print("Tower Scout ready on port 5000...")
    serve(app, host="0.0.0.0", port=5000)
