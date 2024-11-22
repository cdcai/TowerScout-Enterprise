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
# import basic functionality
import sys
import os

# Get the directory of the current script (main.py)
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
import asyncio

from flask import (
    Flask,
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
import os
from shutil import rmtree
import zipfile
import ssl
import time
import tempfile
from PIL import Image, ImageDraw
import threading
import gc
import datetime
import sys
import uuid

from ts_azuremaps import AzureMap

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
    "bing": {"id": "bing", "name": "Bing Maps"},
    "azure": {"id": "azure", "name": "Azure Maps"},
}

# other global variables
google_api_key = ""
bing_api_key = ""
azure_map_key = ""

# prepare uploads directory

uploads_dir = os.path.join(os.getcwd(), "webapp/uploads")
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


# main page route


@app.route("/")
def map_func():

    # for h in request.headers:
    #    print(h)

    # if request.headers.getlist("X-Real-Ip"):
    #    ip = request.headers.getlist("X-Real-Ip")[0]
    # else:
    #    ip = request.remote_addr

    # print("from:", ip)
    # allowed = {"47.215.225.26", "67.188.108.149", "24.126.148.202" } #, "127.0.0.1"}
    # access checks

    # if ip in allowed:
    #    pass
    # elif request.args.get("pw") == "CDC":
    #    pass
    # else:
    #    return send_from_directory('templates', "unauthorized.html")

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
        azure_map_key=azure_map_key,
        dev=dev,
    )


# cache control
# todo: ratchet this up after development


@app.after_request
def add_header(response):
    response.cache_control.max_age = 1
    return response


# # retrieve available engine choices


# @app.route("/getengines")
# def get_engines():
#     print("engines requested")
#     sorted_engines = sorted(engines.items(), key=lambda x: -x[1]["ts"])
#     result = json.dumps([{"id": k, "name": v["name"]} for (k, v) in sorted_engines])
#     return result


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
        print(" session:", id(session))

        # check whether this session is over its limit
        if "tiles" not in session:
            session["tiles"] = 0

        print("tiles queried in session:", session["tiles"])
        if session["tiles"] > MAX_TILES_SESSION:
            return "-1"

        # start time, get params
        start = time.time()
        bounds = request.form.get("bounds")
        engine = request.form.get("engine")
        provider = request.form.get("provider")
        polygons = request.form.get("polygons")
        print("incoming detection request:")
        print(" bounds:", bounds)
        print(" engine:", engine)
        print(" map provider:", provider)
        print(" polygons:", polygons)

        # cropping
        crop_tiles = True

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
            map = AzureMap(azure_map_key)
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
            print()
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
        return tiles
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
    engine = request.form.get("engine")
    print("incoming custom image detection request:")
    print(" engine:", engine)

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
    print()
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
        azure_map_key = f.readline().split()[0]
        bing_api_key = f.readline().split()[0]
        azure_api_key = f.readline().split()[0]
        f.close

    print("Tower Scout ready on port 5000...")
    serve(app, host="0.0.0.0", port=5000)
