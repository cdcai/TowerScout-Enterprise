
// TowerScout
// A tool for identifying cooling towers from satellite and aerial imagery

// TowerScout Team:
// Karen Wong, Gunnar Mein, Thaddeus Segura, Jia Lu

// Licensed under CC-BY-NC-SA-4.0
// (see LICENSE.TXT in the root of the repository for details)


// TowerScout.js
// client-side logic



// maps

// The location of a spot in central NYC
const nyc = [-74.00820558171071, 40.71083794970947];

// main state
let azureMap = null;
let bingMap = null;
let vaAzureMap = null;
let googleMap = null;
let currentMap;
let engines = {};
let currentProvider = null;
let currentUI = null;
let xhr = null;
let currentElement = null;
let currentAddrElement = null;

const input = document.getElementById("search");
const upload = document.getElementById("upload_file");
const detectionsList = document.getElementById("checkBoxes");
const confSlider = document.getElementById("conf");
const reviewCheckBox = document.getElementById("review");
// dynamically adjust confidence of visible predictions
confSlider.oninput = adjustConfidence;
reviewCheckBox.onchange = changeReviewMode;
const DEFAULT_CONFIDENCE = 0.35




// Initialize and add the map
function initBingMap() {
  bingMap = new BingMap();
  currentMap = bingMap;

  // also add change listeners for the UI providers
  // add change listeners for radio buttons
  currentUI = document.uis.uis[1];
  // setMap(currentUI);
  for (let rad of document.uis.uis) {
    rad.addEventListener('change', function () {
      setMap(this);
    });
  }
}

// Initialize and add the map
function initAzureMap() {
  azureMap = new AzureMap();
  currentMap = azureMap;

  // also add change listeners for the UI providers
  // add change listeners for radio buttons
  currentUI = document.uis.uis[0];
  setMap(currentUI);
  for (let rad of document.uis.uis) {
    rad.addEventListener('change', function () {
      setMap(this);
    });
  }
  getazmapTransactioncountjs(2);
}


//
// Abstract Map base class
//

class TSMap {
  getBounds() {
    throw new Error("not implemented")
  }

  getBoundsUrl() {
    let b = this.getBounds();
    return [b[3], b[0], b[1], b[2]].join(","); // assemble in google format w, s, e, n
  }

  setCenter() {
    throw new Error("not implemented")
  }

  getCenter() {
    let b = this.getBounds();
    return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
  }

  getCenterUrl() {
    let c = this.getCenter();
    return c[0] + "," + c[1];
  }

  getZoom() {
    throw new Error("not implemented")
  }

  setZoom(z) {
    throw new Error("not implemented")
  }
  fitCenter() {
    throw new Error("not implemented")
  }

  search(place) {
    throw new Error("not implemented")
  }

  makeMapRect(o) {
    throw new Error("not implemented")
  }

  updateMapRect(o) {
    throw new Error("not implemented")
  }
}



/**
* This is a reusable function that sets the Azure Maps platform domain,
* signs the request, and makes use of any transformRequest set on the map.
* Use like this: `const data = await processRequest(url);`
*/
async function processRequest(url, map) {
  // Replace the domain placeholder to ensure the same Azure Maps is used throughout the app.
  url = url.replace('{azMapsDomain}', atlas.getDomain());

  // Get the authentication details from the map for use in the request.
  var requestParams = map.authentication.signRequest({ url: url });

  // Transform the request.
  var transform = map.getServiceOptions().transformRequest;
  if (transform) requestParams = transform(url);

  const response = await fetch(requestParams.url, {
      method: 'GET',
      mode: 'cors',
      headers: new Headers(requestParams.headers)
  });

  if (!response.ok) {
      throw new Error(`Network response was not ok: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

/**
 * Azure Maps
 */

class AzureMap extends TSMap {
  constructor() {
    super();
    this.map = new atlas.Map('azureMap', {
      center: [nyc[0], nyc[1]], // [longitude, latitude]
      zoom: 19,
      view: 'Auto',
      language: 'en-US',
      authOptions: {
        authType: 'subscriptionKey',
        subscriptionKey: AZURE_MAP_API_KEY
      },
      style: 'satellite'
    });

    /*Add the Style Control to the map*/
    this.map.controls.add(new atlas.control.StyleControl({
      mapStyles: ['road', 'satellite'],
      layout: 'list'
    }), {
      position: 'top-right'
    });

    this.boundaries = [];
    this.drawingManager = null;

    // Listen for the map's ready event
    this.map.events.add('ready', () => {
      // Now it's safe to load drawing tools and add sources/layers
      this.loadDrawingTools();
      let datasource = new atlas.source.DataSource();
      this.map.sources.add(datasource);
      //Add a layer for rendering point data.
      this.map.layers.add(new atlas.layer.SymbolLayer(datasource));

      //Create a jQuery autocomplete UI widget.
      var geocodeServiceUrlTemplate = 'https://{azMapsDomain}/search/fuzzy/json?typeahead=true&api-version=1.0&query={query}&language=en-US&lon={lon}&lat={lat}&countrySet=US&view=Auto';
      $("#search").autocomplete({
          minLength: 4,   //Don't ask for suggestions until atleast 3 characters have been typed. This will reduce costs by not making requests that will likely not have much relevance.
          source: (request, response) => {
              var center = this.map.getCamera().center;

              //Create a URL to the Azure Maps search service to perform the search.
              var requestUrl = geocodeServiceUrlTemplate.replace('{query}', encodeURIComponent(request.term))
                  .replace('{lon}', center[0])    //Use a lat and lon value of the center the map to bais the results to the current map view.
                  .replace('{lat}', center[1])

              processRequest(requestUrl, this.map).then(data => {
                  response(data.results);
              });
          },
          select: (event, ui) => {
              event.preventDefault();
              document.getElementById("search").value = ui.item.address.freeformAddress
              //Remove any previous added data from the map.
              datasource.clear();

              //Create a point feature to mark the selected location.
              // datasource.add(new atlas.data.Feature(new atlas.data.Point([ui.item.position.lon, ui.item.position.lat]), ui.item));
              //TODO: Do not add a point/feature. Instead shade the area if it is a township/postal code... if it is an address then
              // just center on that address

              //Zoom the map into the selected location.
              this.map.setCamera({
                  bounds: [
                      ui.item.viewport.topLeftPoint.lon, ui.item.viewport.btmRightPoint.lat,
                      ui.item.viewport.btmRightPoint.lon, ui.item.viewport.topLeftPoint.lat
                  ],
                  padding: 30
              });
          }
      }).autocomplete("instance")._renderItem = function (ul, item) {
          //Format the displayed suggestion to show the formatted suggestion string.
          var suggestionLabel = item.address.freeformAddress;

          if (item.poi && item.poi.name) {
              suggestionLabel = item.poi.name + ' (' + suggestionLabel + ')';
          }

          return $("<li>")
              .append("<a>" + suggestionLabel + "</a>")
              .appendTo(ul);
      };
    });
  }

  getBoundary(location) {
    // Use the Azure Maps GeoData API to get the boundary
    const geoDataRequestOptions = {
      entityType: 'Neighborhood', // Adjust based on your needs
      location: location
    };

    this.searchService.getPolygon(geoDataRequestOptions)
      .then(data => {
        if (data && data.results && data.results.length > 0) {
          const polygon = new atlas.data.Polygon(data.results[0].geometry.coordinates);
          this.map.sources.add(new atlas.source.DataSource('boundarySource', { data: polygon }));
          this.map.layers.add(new atlas.layer.PolygonLayer('boundarySource', {
            fillColor: 'rgba(0, 0, 255, 0.5)',
            strokeColor: 'blue',
            strokeWidth: 2
          }));
        } else {
          console.log("Could not find boundary.");
        }
      })
      .catch(error => {
        console.error("Error retrieving boundary:", error);
      });
  }

  loadDrawingTools() {
    // Load the DrawingTools module
    this.drawingManager = new atlas.drawing.DrawingManager(this.map, {
        mode: "draw-polygon",
        toolbar: new atlas.control.DrawingToolbar({
          position: 'top-left',
          style: 'light'
      })
    });

    this.map.events.add('drawingcomplete', this.drawingManager, () => { this.retrieveDrawnBoundaries() });
  }

  retrieveDrawnBoundaries() {
    const shapes = this.drawingManager.getSource().shapes;
    const polys = [];

    if (shapes && shapes.length > 0) {
      console.log('Retrieved ' + shapes.length + ' from the drawing manager.');
      for (let s of shapes) {
        const coordinates = s.data.geometry.coordinates[0];
        const points = coordinates.map(coord => [coord[0], coord[1]]);
        polys.push(new PolygonBoundary(points));
      }
    } else {
      console.log('No shapes in the drawing manager.');
    }
    console.log(`Polys: ${polys}`)
    return polys;
  }

  clearShapes() {
    this.drawingManager.clear();
    this.map.sources.clear();
  }

  clearAll() {
    this.clearShapes();
    Detection.resetAll();
  }

  getBounds() {
    const bounds = this.map.getCamera().bounds;
    return [
      bounds.minLongitude,
      bounds.maxLatitude,
      bounds.maxLongitude,
      bounds.minLatitude
    ];
  }

  fitBounds(b) {
    this.map.setCamera({
      bounds: [
        [b[0], b[1]], // Southwest
        [b[2], b[3]]  // Northeast
      ],
      padding: 0,
      zoom: 19
    });
  }

  setCenter(c) {
    this.map.setCamera({
      center: [c[0], c[1]]
    });
  }

  setZoom(z) {
    this.map.setCamera({
      zoom: z
    });
  }

  addBoundary(b) {
    const points = b.points.map(p => new atlas.data.Position(p[0], p[1]));
    const polygon = new atlas.data.Polygon([points]);

    // Add the polygon to the map
    this.map.sources.add(new atlas.source.DataSource('boundarySource', { data: polygon }));
    this.map.layers.add(new atlas.layer.PolygonLayer('boundarySource', {
      fillColor: 'rgba(0, 0, 255, 0.5)',
      strokeColor: 'blue',
      strokeWidth: 2
    }));

    // Store the boundary for later use
    this.boundaries.push(b);
  }

  showBoundaries() {
    // Set map bounds to fit the union of all active boundaries
    if (this.boundaries.length > 0) {
      const bounds = this.boundaries.reduce((acc, b) => {
        const polygonBounds = new atlas.data.BoundingBox(b.points);
        return acc.union(polygonBounds);
      }, new atlas.data.BoundingBox());

      this.map.setCamera({ bounds: bounds, padding: 0 });
    }
  }

  resetBoundaries() {
    this.boundaries = [];
    this.drawingManager?.getSource()?.clear();
  }

  hasShapes() {
    const shapes = this.drawingManager.getSource().shapes;
    return shapes && shapes.length > 0;
  }

  clearShapes() {
    this.drawingManager.getSource().clear();
  }

  getBoundariesStr() {
    return this.boundaries.map(b => b.toString()).join(", ");
  }

  getZoom() {
    return this.map.getCamera().zoom;
  }
}



//
// Bing Maps
//

function drawBoundary(map, results) {
  var locations = results.searchResults;
  locations.forEach(function (location) {
    var polygon = new Microsoft.Maps.Polygon(location.location.boundary, {
      fillColor: 'rgba(0, 255, 0, 0.5)',
      strokeColor: 'green',
      strokeThickness: 2
    });
    map.entities.push(polygon);
  })
}

class BingMap extends TSMap {
  constructor() {
    super();
    this.map = new Microsoft.Maps.Map('#bingMap', {
      center: new Microsoft.Maps.Location(nyc[1], nyc[0]),
      mapTypeId: Microsoft.Maps.MapTypeId.road,
      zoom: 19,
      maxZoom: 21,
      disableStreetside: true,
      // showSearchBar: true
    });
    let bingMap = this.map;
    Microsoft.Maps.loadModule('Microsoft.Maps.AutoSuggest', function () {
      var options = {
          maxResults: 4,
          map: bingMap
      };
      var manager = new Microsoft.Maps.AutosuggestManager(options);
      manager.attachAutosuggest('#search', '#searchBoxContainer', selectedSuggestion);
    });

    document.getElementById('search').addEventListener('change', Search);

    let searchManager;
    Microsoft.Maps.loadModule(['Microsoft.Maps.SpatialDataService',
    'Microsoft.Maps.Search'], function () {
        searchManager = new Microsoft.Maps.Search.SearchManager(bingMap);
    });

    function Search() {
      //Remove all data from the map.
      bingMap.entities.clear();

      //Create the geocode request.
      var geocodeRequest = {
          where: document.getElementById('search').value,
          callback: getBoundary,
          errorCallback: function (e) {
              //If there is an error, alert the user about it.
              //alert("No results found.");
          }
      };
      searchManager.geocode(geocodeRequest);
    }

//     function Search() {
//       //Remove all data from the map.
//       bingMap.entities.clear;

//         var geocodeRequest = {
//           where: document.getElementById('search').value,
//           callback: function (r) {

//               //Add the first result to the map and zoom into it.
//               if (r && r.results && r.results.length > 0) {
//                 if ((r.results[0].entitySubType == "Address") || (r.results[0].entityType == "PostalAddress")){
// var pin = new Microsoft.Maps.Pushpin(r.results[0].location);
//                   bingMap.entities.push(pin);

//                   bingMap.setView({ bounds: r.results[0].bestView });
//                 }

//               }
//               else{
//                 getBoundary;
//               }

//           },
//           errorCallback: function (e) {
//               //If there is an error, alert the user about it.
//               alert("No results found.");
//           }

//     }

//       searchManager.geocode(geocodeRequest);
//     }

      function getBoundary(geocodeResult){
        //Add the first result to the map and zoom into it.
        if (geocodeResult && geocodeResult.results && geocodeResult.results.length > 0) {
            //Zoom into the location.

            bingMap.setView({ bounds: geocodeResult.results[0].bestView });

            //Create the request options for the GeoData API.
            var geoDataRequestOptions = {
                lod: 1,
                getAllPolygons: true
            };

            //Verify that the geocoded location has a supported entity type.
            switch (geocodeResult.results[0].entityType) {
                case "CountryRegion":
                case "AdminDivision1":
                case "AdminDivision2":
                case "Postcode1":
                case "Postcode2":
                case "Postcode3":
                case "Postcode4":
                case "Neighborhood":
                case "PopulatedPlace":
                    geoDataRequestOptions.entityType = geocodeResult.results[0].entityType;
                    break;
                default:
                    //Display a pushpin if GeoData API does not support EntityType.
                    console.log(`GeoData API does not support EntityType ${geocodeResult.results[0].entityType}.`);
                    return;
            }

            //Use the GeoData API manager to get the boundaries of the zip codes.
            Microsoft.Maps.SpatialDataService.GeoDataAPIManager.getBoundary(
              geocodeResult.results[0].location,
              geoDataRequestOptions,
              bingMap,
              function (data) {
                  //Add the polygons to the map.
                  if (data.results && data.results.length > 0) {
                    bingMap.entities.push(data.results[0].Polygons);
                  } else {
                      console.log(`Could not find boundary for ${document.getElementById('search').value}`)
                  }
              }
            );
        }
      }

    function selectedSuggestion(result) {
      bingMap.entities.clear;
      var geocodeRequest;
      if ((result.entitySubType == "Address") || (result.entityType == "PostalAddress")){
        geocodeRequest = {
          where: result.formattedSuggestion,
          callback: function (r) {
              //Add the first result to the map and zoom into it.
              if (r && r.results && r.results.length > 0) {
                  var pin = new Microsoft.Maps.Pushpin(r.results[0].location);
                  bingMap.entities.push(pin);

                  bingMap.setView({ bounds: r.results[0].bestView });
              }
          },
          errorCallback: function (e) {
              //If there is an error, alert the user about it.
              alert("No results found.");
          }
      }
    }
      else{
        geocodeRequest = {
        where: result.title,
        callback: getBoundary,
        errorCallback: function (e) {
            //If there is an error, alert the user about it.
            alert("No results found.");
        }
    };

      }
      searchManager.geocode(geocodeRequest);
    }



    // get view change event to bias place search results
    // Microsoft.Maps.Events.addHandler(this.map, 'viewchangeend', () => googleMap.biasSearchBox());

    // load the spatial math module
    Microsoft.Maps.loadModule('Microsoft.Maps.SpatialMath', () => { });

    // load the DrawingTools module
    this.tools = null;
    this.drawingManager = null;
    let bMap = this;
    Microsoft.Maps.loadModule('Microsoft.Maps.DrawingTools', function () {
      let tools = new Microsoft.Maps.DrawingTools(bMap.map);
      tools.showDrawingManager(function (manager) {
        bMap.drawingManager = manager;
        manager.setOptions({ drawingBarActions: Microsoft.Maps.DrawingTools.DrawingBarAction.polygon });
        Microsoft.Maps.Events.addHandler(manager, 'drawingEnded', function () { console.log('drawingEnded'); });
        Microsoft.Maps.Events.addHandler(manager, 'drawingModeChanged', function () { console.log('drawingModeChanged'); });
        Microsoft.Maps.Events.addHandler(manager, 'drawingStarted', function () { console.log('drawingStarted'); });
      });
    });

    this.boundaries = [];
  }

  getBounds() {
    let locations = [];

    for(var i = 0; i < this.map.entities.getLength(); i++) {
      var entity = this.map.entities.get(i);
      if(entity instanceof Microsoft.Maps.Polygon) {
        var vertices = entity.getLocations();
        locations = locations.concat(vertices);
      }
    }

    if(locations.length > 0) {
      var bounds = Microsoft.Maps.LocationRect.fromLocations(locations);
      this.map.setView({ bounds: bounds });
    }

    let rect = this.map.getBounds();
    return [
      rect.center.longitude - rect.width / 2,
      rect.center.latitude + rect.height / 2,
      rect.center.longitude + rect.width / 2,
      rect.center.latitude - rect.height / 2
    ];
  }

  fitBounds(b) {
    let locs = [
      new Microsoft.Maps.Location(b[1], b[0]),
      new Microsoft.Maps.Location(b[3], b[2]),
    ];
    let rect = Microsoft.Maps.LocationRect.fromLocations(locs);
    this.map.setView({ bounds: rect, padding: 0, zoom: 19 });
  }

  setCenter(c) {
    this.map.setView({
      center: new Microsoft.Maps.Location(c[1], c[0]),
    });
  }

  setZoom(z) {
    this.map.setView({
      zoom: z
    });
  }


  makeMapRect(o, listener) {
    let locs = [
      new Microsoft.Maps.Location(o.y1, o.x1),
      new Microsoft.Maps.Location(o.y1, o.x2),
      new Microsoft.Maps.Location(o.y2, o.x2),
      new Microsoft.Maps.Location(o.y2, o.x1),
      new Microsoft.Maps.Location(o.y1, o.x1)
    ];
    let color = Microsoft.Maps.Color.fromHex(o.color);
    color.a = o.opacity;
    let polygon = new Microsoft.Maps.Polygon(
      locs,
      {
        fillColor: color,
        strokeColor: o.color,
        strokeThickness: 1
      });

    if (typeof listener !== 'undefined') {
      Microsoft.Maps.Events.addHandler(polygon, 'click', listener);
    }
    this.map.entities.push(polygon);
    return polygon;
  }

  colorMapRect(o, color) {
    let fcolor = Microsoft.Maps.Color.fromHex(color);
    fcolor.a = o.opacity;
    o.mapRect.setOptions({ strokeColor: color, fillColor: fcolor });
  }

  updateMapRect(o, onoff) {
    let r = o.mapRect;
    r.setOptions({ visible: onoff });
  }

  getZoom() {
    return this.map.getZoom();
  }

  resetBoundaries() {
    this.drawingManager.clear()
    for (let b of this.boundaries) {
      for (let i = this.map.entities.getLength() - 1; i >= 0; i--) {
        let obj = this.map.entities.get(i);
        // if (obj === b.bingObject) {
          this.map.entities.removeAt(i);
        // }
      }
      b.bingObject = null;
    }
    this.boundaries = [];
    this.map.entities.clear();
  }

  addBoundary(b) {
    // make BingMap objects and link to them
    // all boundaries are polygons
    let points = [];
    for (let p of b.points) {
      points.push(new Microsoft.Maps.Location(p[1], p[0]));
    }
    const poly = new Microsoft.Maps.Polygon(points, {
      fillColor: "rgba(0,0,0,0)",
      strokeColor: "#0000FF",
      strokeThickness: 2
    });
    this.map.entities.push(poly);
    b.bingObject = poly;
    b.bingObjectBounds = poly.geometry.boundingBox;

    // add to active bounds
    this.boundaries.push(b);
  }

  showBoundaries() {
    // set map bounds to fit union of all active boundaries
    let bobjs = this.boundaries.map(x => x.bingObject);
    let bounds = Microsoft.Maps.LocationRect.fromShapes(bobjs);
    this.map.setView({ bounds: bounds, padding: 0 });
  }

  retrieveDrawnBoundaries() {
    let shapes = this.drawingManager.getPrimitives();
    let polys = [];

    if (shapes && shapes.length > 0) {
      console.log('Retrieved ' + shapes.length + ' from the drawing manager.');
      for (let s of shapes) {
        console.log("Adding polygon" + s.geometry.bounds.toString());
        let x = s.geometry.rings[0].x;
        let y = s.geometry.rings[0].y;
        let points = []
        for (let i = 0; i < x.length; i++) {
          points.push([x[i], y[i]]);
        }
        polys.push(new PolygonBoundary(points))
      }
    } else {
      console.log('No shapes in the drawing manager.');
    }

    return polys;
  }

  hasShapes() {
    let shapes = this.drawingManager.getPrimitives();
    return shapes && shapes.length > 0;
  }

  addShapes() {
    let shapes = this.drawingManager.getPrimitives();

    if (shapes && shapes.length > 0) {
      console.log('Retrieved ' + shapes.length + ' from the drawing manager.');
      for (let s of shapes) {
        console.log("Adding " + s.geometry.bounds.toString());
        let x1 = s.geometry.boundingBox.getWest();
        let y1 = s.geometry.boundingBox.getNorth();
        let x2 = s.geometry.boundingBox.getEast();
        let y2 = s.geometry.boundingBox.getSouth();

        let tileIds = Tile.getTileIds(x1, y1, x2, y2);
        for (let tileId of tileIds) {
          let tile = Tile_tiles[tileId]
          x1 = Math.max(x1, tile.x1);
          x1 = Math.min(x1, tile.x2);
          x2 = Math.max(x2, tile.x1);
          x2 = Math.min(x2, tile.x2);
          y1 = Math.max(y1, tile.y2);
          y1 = Math.min(y1, tile.y1);
          y2 = Math.max(y2, tile.y2);
          y2 = Math.min(y2, tile.y1);
          let det = new Detection(x1, y1, x2, y2,
            'added', 1.0, tileId, -1 /*id_in_tile*/, true, true);
          det.update();
        }

        augmentDetections();
      }
      this.drawingManager.clear();
      // this.map.entities.clear();
    } else {
      console.log('No shapes in the drawing manager.');
    }
  }

  clearShapes() {
    this.drawingManager.clear();
    this.map.entities.clear();
  }

  clearAll() {
    this.clearShapes();
    // now, also go through Detection_detections and take out the blue ones
    // let dets = [];
    // for (let det of Detection_detections) {
    //   if (det.conf !== 1.0) {
    //     det.id = dets.length;
    //     dets.push(det);
    //   }
    // }

    // Detection_detections = dets;
    // Detection.generateList();
    Detection.resetAll();
  }

  getBoundariesStr() {
    let result = [];
    for (let b of this.boundaries) {
      result.push(b.toString())
    }
    return "[" + result.join(",") + "]";
  }

}

class clsAzureMap extends TSMap {
  constructor() {
    super();
    this.map = new new atlas.Map('#azureMap', {
      center: new Microsoft.Maps.Location(nyc[1], nyc[0]),
      mapTypeId: Microsoft.Maps.MapTypeId.road,
      zoom: 19,
      maxZoom: 21,
      disableStreetside: true,
      View: 'Auto'
      // showSearchBar: true
    });
    let azureMap = this.map;
    Microsoft.Maps.loadModule('Microsoft.Maps.AutoSuggest', function () {
      var options = {
          maxResults: 4,
          map: bingMap
      };
      var manager = new Microsoft.Maps.AutosuggestManager(options);
      manager.attachAutosuggest('#search', '#searchBoxContainer', selectedSuggestion);
    });

    document.getElementById('search').addEventListener('change', Search);

    let searchManager;
    Microsoft.Maps.loadModule(['Microsoft.Maps.SpatialDataService',
    'Microsoft.Maps.Search'], function () {
        searchManager = new Microsoft.Maps.Search.SearchManager(bingMap);
    });

    function Search() {
      //Remove all data from the map.
      bingMap.entities.clear();

      //Create the geocode request.
      var geocodeRequest = {
          where: document.getElementById('search').value,
          callback: getBoundary,
          errorCallback: function (e) {
              //If there is an error, alert the user about it.
              //alert("No results found.");
          }
      };
      searchManager.geocode(geocodeRequest);
    }

//     function Search() {
//       //Remove all data from the map.
//       bingMap.entities.clear;

//         var geocodeRequest = {
//           where: document.getElementById('search').value,
//           callback: function (r) {

//               //Add the first result to the map and zoom into it.
//               if (r && r.results && r.results.length > 0) {
//                 if ((r.results[0].entitySubType == "Address") || (r.results[0].entityType == "PostalAddress")){
// var pin = new Microsoft.Maps.Pushpin(r.results[0].location);
//                   bingMap.entities.push(pin);

//                   bingMap.setView({ bounds: r.results[0].bestView });
//                 }

//               }
//               else{
//                 getBoundary;
//               }

//           },
//           errorCallback: function (e) {
//               //If there is an error, alert the user about it.
//               alert("No results found.");
//           }

//     }

//       searchManager.geocode(geocodeRequest);
//     }

      function getBoundary(geocodeResult){
        //Add the first result to the map and zoom into it.
        if (geocodeResult && geocodeResult.results && geocodeResult.results.length > 0) {
            //Zoom into the location.

            bingMap.setView({ bounds: geocodeResult.results[0].bestView });

            //Create the request options for the GeoData API.
            var geoDataRequestOptions = {
                lod: 1,
                getAllPolygons: true
            };

            //Verify that the geocoded location has a supported entity type.
            switch (geocodeResult.results[0].entityType) {
                case "CountryRegion":
                case "AdminDivision1":
                case "AdminDivision2":
                case "Postcode1":
                case "Postcode2":
                case "Postcode3":
                case "Postcode4":
                case "Neighborhood":
                case "PopulatedPlace":
                    geoDataRequestOptions.entityType = geocodeResult.results[0].entityType;
                    break;
                default:
                    //Display a pushpin if GeoData API does not support EntityType.
                    console.log(`GeoData API does not support EntityType ${geocodeResult.results[0].entityType}.`);
                    return;
            }

            //Use the GeoData API manager to get the boundaries of the zip codes.
            Microsoft.Maps.SpatialDataService.GeoDataAPIManager.getBoundary(
              geocodeResult.results[0].location,
              geoDataRequestOptions,
              bingMap,
              function (data) {
                  //Add the polygons to the map.
                  if (data.results && data.results.length > 0) {
                    bingMap.entities.push(data.results[0].Polygons);
                  } else {
                      console.log(`Could not find boundary for ${document.getElementById('search').value}`)
                  }
              }
            );
        }
      }

    function selectedSuggestion(result) {
      bingMap.entities.clear;
      var geocodeRequest;
      if ((result.entitySubType == "Address") || (result.entityType == "PostalAddress")){
        geocodeRequest = {
          where: result.formattedSuggestion,
          callback: function (r) {
              //Add the first result to the map and zoom into it.
              if (r && r.results && r.results.length > 0) {
                  var pin = new Microsoft.Maps.Pushpin(r.results[0].location);
                  bingMap.entities.push(pin);

                  bingMap.setView({ bounds: r.results[0].bestView });
              }
          },
          errorCallback: function (e) {
              //If there is an error, alert the user about it.
              alert("No results found.");
          }
      }
    }
      else{
        geocodeRequest = {
        where: result.title,
        callback: getBoundary,
        errorCallback: function (e) {
            //If there is an error, alert the user about it.
            alert("No results found.");
        }
    };

      }
      searchManager.geocode(geocodeRequest);
    }



    // get view change event to bias place search results
    // Microsoft.Maps.Events.addHandler(this.map, 'viewchangeend', () => googleMap.biasSearchBox());

    // load the spatial math module
    Microsoft.Maps.loadModule('Microsoft.Maps.SpatialMath', () => { });

    // load the DrawingTools module
    this.tools = null;
    this.drawingManager = null;
    let bMap = this;
    Microsoft.Maps.loadModule('Microsoft.Maps.DrawingTools', function () {
      let tools = new Microsoft.Maps.DrawingTools(bMap.map);
      tools.showDrawingManager(function (manager) {
        bMap.drawingManager = manager;
        manager.setOptions({ drawingBarActions: Microsoft.Maps.DrawingTools.DrawingBarAction.polygon });
        Microsoft.Maps.Events.addHandler(manager, 'drawingEnded', function () { console.log('drawingEnded'); });
        Microsoft.Maps.Events.addHandler(manager, 'drawingModeChanged', function () { console.log('drawingModeChanged'); });
        Microsoft.Maps.Events.addHandler(manager, 'drawingStarted', function () { console.log('drawingStarted'); });
      });
    });

    this.boundaries = [];
  }

  getBounds() {
    let locations = [];

    for(var i = 0; i < this.map.entities.getLength(); i++) {
      var entity = this.map.entities.get(i);
      if(entity instanceof Microsoft.Maps.Polygon) {
        var vertices = entity.getLocations();
        locations = locations.concat(vertices);
      }
    }

    if(locations.length > 0) {
      var bounds = Microsoft.Maps.LocationRect.fromLocations(locations);
      this.map.setView({ bounds: bounds });
    }

    let rect = this.map.getBounds();
    return [
      rect.center.longitude - rect.width / 2,
      rect.center.latitude + rect.height / 2,
      rect.center.longitude + rect.width / 2,
      rect.center.latitude - rect.height / 2
    ];
  }

  fitBounds(b) {
    let locs = [
      new Microsoft.Maps.Location(b[1], b[0]),
      new Microsoft.Maps.Location(b[3], b[2]),
    ];
    let rect = Microsoft.Maps.LocationRect.fromLocations(locs);
    this.map.setView({ bounds: rect, padding: 0, zoom: 19 });
  }

  setCenter(c) {
    this.map.setView({
      center: new Microsoft.Maps.Location(c[1], c[0]),
    });
  }

  setZoom(z) {
    this.map.setView({
      zoom: z
    });
  }


  makeMapRect(o, listener) {
    let locs = [
      new Microsoft.Maps.Location(o.y1, o.x1),
      new Microsoft.Maps.Location(o.y1, o.x2),
      new Microsoft.Maps.Location(o.y2, o.x2),
      new Microsoft.Maps.Location(o.y2, o.x1),
      new Microsoft.Maps.Location(o.y1, o.x1)
    ];
    let color = Microsoft.Maps.Color.fromHex(o.color);
    color.a = o.opacity;
    let polygon = new Microsoft.Maps.Polygon(
      locs,
      {
        fillColor: color,
        strokeColor: o.color,
        strokeThickness: 1
      });

    if (typeof listener !== 'undefined') {
      Microsoft.Maps.Events.addHandler(polygon, 'click', listener);
    }
    this.map.entities.push(polygon);
    return polygon;
  }

  colorMapRect(o, color) {
    let fcolor = Microsoft.Maps.Color.fromHex(color);
    fcolor.a = o.opacity;
    o.mapRect.setOptions({ strokeColor: color, fillColor: fcolor });
  }

  updateMapRect(o, onoff) {
    let r = o.mapRect;
    r.setOptions({ visible: onoff });
  }

  getZoom() {
    return this.map.getZoom();
  }

  resetBoundaries() {
    for (let b of this.boundaries) {
      for (let i = this.map.entities.getLength() - 1; i >= 0; i--) {
        let obj = this.map.entities.get(i);
        // if (obj === b.bingObject) {
          this.map.entities.removeAt(i);
        // }
      }
      b.bingObject = null;
    }
    this.boundaries = [];
    this.map.entities.clear();
  }

  addBoundary(b) {
    // make BingMap objects and link to them
    // all boundaries are polygons
    let points = [];
    for (let p of b.points) {
      points.push(new Microsoft.Maps.Location(p[1], p[0]));
    }
    const poly = new Microsoft.Maps.Polygon(points, {
      fillColor: "rgba(0,0,0,0)",
      strokeColor: "#0000FF",
      strokeThickness: 2
    });
    this.map.entities.push(poly);
    b.bingObject = poly;
    b.bingObjectBounds = poly.geometry.boundingBox;

    // add to active bounds
    this.boundaries.push(b);
  }

  showBoundaries() {
    // set map bounds to fit union of all active boundaries
    let bobjs = this.boundaries.map(x => x.bingObject);
    let bounds = Microsoft.Maps.LocationRect.fromShapes(bobjs);
    this.map.setView({ bounds: bounds, padding: 0 });
  }

  retrieveDrawnBoundaries() {
    let shapes = this.drawingManager.getPrimitives();
    let polys = [];

    if (shapes && shapes.length > 0) {
      console.log('Retrieved ' + shapes.length + ' from the drawing manager.');
      for (let s of shapes) {
        console.log("Adding polygon" + s.geometry.bounds.toString());
        let x = s.geometry.rings[0].x;
        let y = s.geometry.rings[0].y;
        let points = []
        for (let i = 0; i < x.length; i++) {
          points.push([x[i], y[i]]);
        }
        polys.push(new PolygonBoundary(points))
      }
    } else {
      console.log('No shapes in the drawing manager.');
    }

    return polys;
  }

  hasShapes() {
    let shapes = this.drawingManager.getPrimitives();
    return shapes && shapes.length > 0;
  }

  addShapes() {
    let shapes = this.drawingManager.getPrimitives();

    if (shapes && shapes.length > 0) {
      console.log('Retrieved ' + shapes.length + ' from the drawing manager.');
      for (let s of shapes) {
        console.log("Adding " + s.geometry.bounds.toString());
        let x1 = s.geometry.boundingBox.getWest();
        let y1 = s.geometry.boundingBox.getNorth();
        let x2 = s.geometry.boundingBox.getEast();
        let y2 = s.geometry.boundingBox.getSouth();

        let tileIds = Tile.getTileIds(x1, y1, x2, y2);
        for (let tileId of tileIds) {
          let tile = Tile_tiles[tileId]
          x1 = Math.max(x1, tile.x1);
          x1 = Math.min(x1, tile.x2);
          x2 = Math.max(x2, tile.x1);
          x2 = Math.min(x2, tile.x2);
          y1 = Math.max(y1, tile.y2);
          y1 = Math.min(y1, tile.y1);
          y2 = Math.max(y2, tile.y2);
          y2 = Math.min(y2, tile.y1);
          let det = new Detection(x1, y1, x2, y2,
            'added', 1.0, tileId, -1 /*id_in_tile*/, true, true);
          det.update();
        }

        augmentDetections();
      }
      this.drawingManager.clear();
      // this.map.entities.clear();
    } else {
      console.log('No shapes in the drawing manager.');
    }
  }

  clearShapes() {
    this.drawingManager.clear();
    this.map.entities.clear();
  }

  clearAll() {
    this.clearShapes();
    // now, also go through Detection_detections and take out the blue ones
    // let dets = [];
    // for (let det of Detection_detections) {
    //   if (det.conf !== 1.0) {
    //     det.id = dets.length;
    //     dets.push(det);
    //   }
    // }

    // Detection_detections = dets;
    // Detection.generateList();
    Detection.resetAll();
  }

  getBoundariesStr() {
    let result = [];
    for (let b of this.boundaries) {
      result.push(b.toString())
    }
    return "[" + result.join(",") + "]";
  }

}
//
// boundaries: simple, circle, polygon
//

class Boundary {
  constructor(kind) {
    this.kind = kind;
  }

  toString() {
    throw new Error("not implemented");
  }
}

class PolygonBoundary extends Boundary {
  constructor(points) {
    super("polygon");
    this.points = points;
  }

  toString() {
    return '{"kind":"polygon","points":' + JSON.stringify(this.points) + '}';
  }
}

class SimpleBoundary extends PolygonBoundary {
  constructor(bounds) {
    super("simple:" + bounds);
    this.points = [[bounds[0], bounds[1]],
    [bounds[2], bounds[1]],
    [bounds[2], bounds[3]],
    [bounds[0], bounds[3]],
    [bounds[0], bounds[1]]
    ];
  }
}

class CircleBoundary extends PolygonBoundary {
  constructor(center, radius) {
    super("circle: " + center + ", " + radius + " m");
    // use MSFT stuff to compute the circle
    let locs = Microsoft.Maps.SpatialMath.getRegularPolygon(
      new Microsoft.Maps.Location(center[1], center[0]),
      radius,
      256,
      Microsoft.Maps.SpatialMath.DistanceUnits.Meters);
    this.points = locs.map(l => [l.longitude, l.latitude]);
  }
}



//
// PlaceRects - rectangles on the map (results, tiles, bounding boxes)
//

class PlaceRect {

  constructor(x1, y1, x2, y2, color, fillColor, opacity, classname, listener) {
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
    this.color = color;
    this.fillColor = fillColor;
    this.opacity = opacity;
    this.classname = classname;
    this.address = "<unknown address>";
    this.map = currentMap
    this.mapRect = this.map.makeMapRect(this, listener);
    this.update();
    this.listener = listener;
  }

  centerInMap() {
    // this.map.setCenter([(this.x1 + this.x2) / 2, (this.y1 + this.y2) / 2]);
    // currentMap.setZoom(19);
    // googleMap.setCenter([(this.x1 + this.x2) / 2, (this.y1 + this.y2) / 2]);
    // googleMap.setZoom(19);
    bingMap.setCenter([(this.x1 + this.x2) / 2, (this.y1 + this.y2) / 2]);
    bingMap.setZoom(19);
  }

  getCenter() {
    return [(this.x1 + this.x2) / 2, (this.y1 + this.y2) / 2];
  }

  getCenterUrl() {
    let c = this.getCenter();
    return c[1] + "," + c[0];
  }

  augment(addr) {
    this.addrSpan.innerText = addr;
    this.address = addr;
    //console.log("tower " + i + ": " + addr)
  }

  highlight(color) {
    currentMap.colorMapRect(this, color);
    setTimeout(() => {
      currentMap.colorMapRect(this, this.color);
    }, 5000);
  }

  update(newMap) {
    if (typeof newMap !== 'undefined') {
      this.map.updateMapRect(this, false);
      this.mapRect = newMap.makeMapRect(this, this.listener);
      this.map = newMap;
    }
    this.map.updateMapRect(this, true);
  }
}

let Tile_tiles = [];
class Tile extends PlaceRect {
  static resetAll() {
    for (let tile of Tile_tiles) {
      currentMap.updateMapRect(tile, false);
    }
    Tile_tiles = [];
  }

  constructor(x1, y1, x2, y2, metadata, url) {
    super(x1, y1, x2, y2, "#0000FF", "#0000FF", 0.0, "tile")
    this.metadata = metadata; // for Bing maps
    this.url = url

    Tile_tiles.push(this);
  }

  // find the ids for all tiles that the center of this box belongs to
  static getTileIds(x1, y1, x2, y2) {
    let result = [];
    for (let i = 0; i < Tile_tiles.length; i++) {
      let t = Tile_tiles[i]
      // compute center
      let cx = (x1 + x2) / 2;
      let cy = (y1 + y2) / 2;

      // check if center in tile
      if (cx >= t.x1 && cx <= t.x2 && cy <= t.y1 && cy >= t.y2) {
        result.push(i);
      }
    }
    return result;
  }

  // tile navigation for review pane
  static number() {
    let index = document.getElementById("tile").value;
    if (index === "") {
      index = "0";
    } else {
      index = Number(index) % Tile_tiles.length;
    }
    document.getElementById("tile").value = String(index);
    Tile_tiles[index].centerInMap();
  }

  static prev() {
    let index = document.getElementById("tile").value;
    if (index === "") {
      index = "0";
    } else {
      index = (((Number(index) - 1) % Tile_tiles.length) + Tile_tiles.length) % Tile_tiles.length; // don't ask
    }
    document.getElementById("tile").value = String(index);
    Tile_tiles[index].centerInMap();
  }

  static next() {
    let index = document.getElementById("tile").value;
    if (index === "") {
      index = "0";
    } else {
      index = (Number(index) + 1) % Tile_tiles.length;
    }
    document.getElementById("tile").value = String(index);
    Tile_tiles[index].centerInMap();
  }
}


let Detection_detections = []
let Detection_detectionsAugmented = 0;
let Detection_minConfidence = DEFAULT_CONFIDENCE;
let Detection_current = null;

class Detection extends PlaceRect {
  static resetAll() {
    for (let det of Detection_detections) {
      det.select(false);
    }
    Detection_detections = [];
    Detection_detectionsAugmented = 0;
    detectionsList.innerHTML = "";
  }

  constructor(x1, y1, x2, y2, classname, conf, tile, idInTile, inside, selected, secondary) {
    super(x1, y1, x2, y2, conf === 1.0 ? "blue" : "#FF0000", conf === 1.0 ? "blue" : "#FF0000", 0.2, classname, () => {
      this.highlight(false, true);
    })
    this.conf = conf;
    this.inside = inside;
    this.idInTile = idInTile;
    this.selected = selected;
    this.address = "";
    this.maxConf = conf; // minimum confidence across same address towers, only recorded in first
    this.firstDet = null; // first of block of same address towers
    this.tile = tile; // id of detection tile
    this.secondary = secondary;

    this.id = Detection_detections.length;
    this.originalId = this.id;
    //console.log("Detection #" + this.id + " is " + (this.selected ? "" : "not ") + "selected");
    Detection_detections.push(this);
  }

  static sort() {
    Detection_detections.sort((a, b) => {
      if (a.address < b.address) {
        return -1;
      } else if (a.address > b.address) {
        return 1;
      } else {
        return b.conf - a.conf;
      }
    });

    // fix ids
    for (let i = 0; i < Detection_detections.length; i++) {
      let det = Detection_detections[i];
      det.id = i;
    }
  }
  static generateList() {
    let currentAddr = "";
    let firstDet = null;
    let boxes = "<ul>";
    let count = 0;
    for (let det of Detection_detections) {
      if (det.address !== currentAddr) {
        if (currentAddr !== "") {
          boxes += "</ul></li>";
        }
        boxes += "<li id='addrli" + det.id + "'>";
        boxes += "<span class='caret' onclick='";
        boxes += "this.parentElement.querySelector(\".nested\").classList.toggle(\"active\"),";
        boxes += "this.classList.toggle(\"caret-down\")';"
        boxes += "'></span>";
        boxes += "<label id='lbladdrcb" + det.id + "' style='display:none' for='addrcb" + det.id + "'>addrcb"+det.id + "</label>"
        boxes += "<input aria-labelledby = 'lbladdrcb" + det.id + "' type='checkbox' id='addrcb" + det.id + "' name='addrcb" + det.id;
        boxes + "' value='";
        boxes += det.id + "' checked style='display:inline;vertical-align:-10%;'"
        boxes += " onclick='Detection_detections[" + det.id + "].selectAddr(this.checked)'>";
        boxes += "<span class='address' id='addrlabel" + det.id + "'";
        boxes += " onclick='Detection.showDetection(" + det.id + ", true)'>"
        boxes += det.address + "</span><br>";
        boxes += "<ul class='nested' id='towerslist" + det.id;
        boxes += "' style='text-indent:-25px; padding-left: 60px;'>";
        currentAddr = det.address;
        firstDet = det;
      }
      boxes += det.generateCheckBox();
      firstDet.maxConf = Math.max(det.conf, firstDet.maxConf); // record min conf in block header
      firstDet.maxP2 = Math.max(det.p2, firstDet.maxP2)
      det.firstDet = firstDet; // record block header
      det.indexInList = count;
      det.update();
      count++;
    }
    boxes += "</li></ul>";
    detectionsList.innerHTML = boxes;
    // Call the function
    assignAriaLabels();
  }

  generateCheckBox() {
    let meta = Tile_tiles[this.tile].metadata;
    let p2 = (this.secondary > 0 && this.secondary < 1.0 ? ",&nbsp;P2(" + this.secondary.toFixed(2) + ")" : "")
    let box = "<li><div style='display:block' id='detdiv" + this.id + "'>";
    box += "<label id ='lbldetcb" + this.id + "' style='display:none' for='detcb" + this.id + "'>lbl" + this.id + "</label>"
    box += "<input aria-labelledby = 'lbldetcb" + this.id + "' type='checkbox' id='detcb" + this.id + "' name='detcb" + this.id + "'";
    box += " value='" + this.id + "' " + (this.selected ? "checked" : "");
    box += " style='display:inline;vertical-align:-10%;'"
    box += " onclick='Detection_detections[" + this.id + "].select(undefined)'>";
    box += "&nbsp;";
    box += "<span class='address' onclick='Detection.showDetection(" + this.id + ", true)' ";
    box += "id='plabel" + this.id + "'>";
    box += "P(" + this.conf.toFixed(2) + ")" + p2 + (meta !== "" ? ",&nbsp" + meta : "") + "</span></li>";
    box += "</div>";

    this.checkBoxId = 'detdiv' + this.id;
    this.labelId = 'plabel' + this.id;
    return box;
  }



  select(onoff) {
    if (typeof onoff === 'undefined') {
      onoff = !this.selected;
    }
    this.selected = onoff;
    document.getElementById("detcb" + this.id).checked = onoff;
    this.update();
  }

  selectAddr(onoff) {
    if (typeof onoff === 'undefined') {
      onoff = !this.selected;
    }
    for (let det of Detection_detections) {
      if (det.address === this.address) {
        det.selected = onoff;
        document.getElementById("detcb" + det.id).checked = onoff;
        det.update();
      }
    }
  }

  show(onoff) {
    document.getElementById("detdiv" + this.id).style.display = onoff ? "block" : "none";
  }

  isShown() {
    return document.getElementById("detdiv" + this.id).style.display === "block";
  }

  showAddr(onoff) {
    document.getElementById("addrli" + this.id).style.display = onoff ? "block" : "none";
  }

  static showDetection(id, center) {
    Detection_detections[id].highlight(center, false);
  }

  highlight(center, scroll) {
    let firstDet = this.firstDet;

    if (currentAddrElement !== null) {
      currentAddrElement.style.fontWeight = "normal";
      currentAddrElement.style.textDecoration = "";
      currentElement.style.fontWeight = "normal";
      currentElement.style.textDecoration = "";
    }

    // highlight the address
    let element = document.getElementById('addrlabel' + firstDet.id);
    element.style.fontWeight = "bolder";
    element.style.textDecoration = "underline";
    currentAddrElement = element;

    // make sure parent element is open
    element.parentNode.firstChild.classList.add('caret-down');
    // and list displayed
    element.parentNode.lastChild.classList.add('active');

    // highlight the individual detection
    element = document.getElementById(this.labelId);
    if (scroll) {
      currentAddrElement.scrollIntoView();
    }
    element.style.fontWeight = "bolder";
    element.style.textDecoration = "underline";
    currentElement = element;
    document.getElementById("detection").value = this.indexInList;


    if (center) {
      this.centerInMap();
    }

    if (Detection_current !== null) {
      Detection_current.resetHighlight();
    }
    super.highlight("green");
    Detection_current = this;
  }

  resetHighlight() {
    super.highlight(this.color);
  }


  augment(addr) {
    // this.addrSpan.innerText = addr;
    this.address = addr;
    Detection_detectionsAugmented++;
    //console.log("tower " + i + ": " + addr)
  }

  update(newMap) {
    // first, process any map UI change
    super.update(newMap)

    let meetsInside = reviewCheckBox.checked || this.inside;
    // then update by confidence
    this.map.updateMapRect(this, this.selected && this.conf >= Detection_minConfidence && meetsInside);
  }

  // navigation for review pane
  static number() {
    let index = document.getElementById("detection").value;
    if (index === "") {
      index = "0";
    } else {
      index = Number(index);
    }
    document.getElementById("detection").value = String(this.navigateTo(index));
  }

  // navigation for review pane
  static prev() {
    let index = document.getElementById("detection").value;
    if (index === "") {
      index = "0";
    } else {
      index = Number(index) - 1;
    }
    document.getElementById("detection").value = String(this.navigateTo(index));
  }

  static next() {
    let index = document.getElementById("detection").value;
    if (index === "") {
      index = "0";
    } else {
      index = Number(index) + 1;
    }
    document.getElementById("detection").value = String(this.navigateTo(index));
  }

  static navigateTo(index) {
    // first, count shown detections
    let count = 0;
    for (let det of Detection_detections) {
      if (det.isShown() && det.selected) {
        count++;
      }
    }
    // limit index to count
    index = ((index % count) + count) % count;

    // now find and center
    let j = 0;
    for (let det of Detection_detections) {
      if (det.isShown() && det.selected) {
        if (j == index) {
          det.highlight(true, true);
          return index;
        }
        j++;
      }
    }
    return index;
  }
}


function createElementFromHTML(htmlString) {
  let div = document.createElement('div');
  div.innerHTML = htmlString.trim();
  return div.firstChild;
}

// retrieve satellite image and detect objects
function getObjects(estimate) {
  //let center = currentMap.getCenterUrl();

  if (Detection_detections.length > 0) {
    if (!window.confirm("This will erase current detections. Proceed?")) {
       // erase the previous set of towers and tiles
       this.clearAll();
  return;
}
  }

  let engine = $('input[name=model]:checked', '#engines').val()
  let provider = $('input[name=provider]:checked', '#providers').val()
  provider = provider.substring(0, provider.length - 9);
  // let boundaries = googleMap.getBoundariesStr();
  // if (boundaries === "[]" && radius == "") {
  //   console.log("No boundary selected, instead using viewport: " + googleMap.getBounds())
  //   googleMap.addBoundary(new SimpleBoundary(googleMap.getBounds()));
  //   bingMap.addBoundary(new SimpleBoundary(googleMap.getBounds()));
  // }


  // now get the boundaries ready to ship
  let bounds = currentMap.getBoundsUrl();

  if (currentMap.boundaries.length === 0) {
    if (currentMap.hasShapes()) {
      drawnBoundary();
    }
  }

  let boundaries = currentMap.getBoundariesStr();
  let kinds = ["None", "Polygon", "Multiple polygons"]
  if (estimate) {
    console.log("Estimate request in progress");
  } else {
    console.log("Detection request in progress");
  }

  // erase the previous set of towers and tiles
  Detection.resetAll();
  Tile.resetAll();

  // first, play the request, but get an estimate of the number of tiles
  const formData = new FormData();
  formData.append('bounds', bounds);
  formData.append('engine',engine);
  formData.append('provider',provider );
  formData.append('polygons', boundaries);
  formData.append('estimate', "yes");

  fetch("/getobjects",  { method: "POST", body: formData, })
    .then(result => result.text())
    .then(result => {
      if (Number(result) === -1) {
        fatalError("Tile limit for this session exceeded. Please close browser to continue.")
        return;
      }
      console.log("Number of tiles: " + result + ", estimated time: "
        + (Math.round(Number(result) * secsPerTile * 10) / 10) + " s");
        // Get from Dev  Key Vault by default. Need to create secrets in the Prod Key Vault

      // let nt = estimateNumTiles(currentMap.getZoom());
      // console.log("  Estimated tiles:" + nt);
      if (estimate) {
        return;
      }

      // actual retrieval process starts here
      let nt = Number(result);
      enableProgress(nt);
      setProgress(0);
      let startTime = performance.now();

      // now, the actual request

      Detection.resetAll();
      formData.delete("estimate");
      fetch("/getobjects", { method: "POST", body: formData })
        .then(response => response.json())
        .then(result => {
          // Need to add code to process the results from EDAV
          disableProgress(0, 0);
          return;
          // processObjects(result, startTime);
        })
        .catch(e => {
          console.log(e + ": "); disableProgress(0, 0);
        });
    });

    getazmapTransactioncountjs(2);
}

//get Azure map transaction count for the current month
function getazmapTransactioncountjs(intEnv)
{
  param = intEnv
  // fetch('/getazmaptransactions', { method: "POST", body:  JSON.stringify({ param: param }) })
  fetch('/getazmaptransactions')
    .then(response => {

      return response.text();
    })
    .then(data => {
      // Assign the response value to a variable
      const message = data;
      // console.log(message); // Log the value to the console
      document.getElementById('lblazmaptransactions').innerText = message; // Display the string in the div
  })
    .catch(error => {
      console.log(error);
    });

}

function processObjects(result, startTime) {
  conf = Number(document.getElementById("conf").value);
  if (result.length === 0) {
    console.log(":::::: Area too big. Please " + (radius !== "" ? "enter a smaller radius." : "zoom in."));
    disableProgress(0, 0);
    return;
  }

  // make detection objects
  for (let r of result) {
    if (r['class'] === 0) {
      let det = new Detection(r['x1'], r['y1'], r['x2'], r['y2'],
        r['class_name'], r['conf'], r['tile'], r['id_in_tile'], r['inside'], r['selected'], r['secondary']);
    } else if (r['class'] === 1) {
      let tile = new Tile(r['x1'], r['y1'], r['x2'], r['y2'], r['metadata'], r['url']);
    }
  }
  //console.log("" + Detection_detections.length + " detections.")
  disableProgress((performance.now() - startTime) / 1000, Tile_tiles.length);
  augmentDetections();
  // if (boundaries != "[]") {
  //   googleMap.showBoundaries();
  //   bingMap.showBoundaries();
  // }
}

function cancelRequest() {
  xhr.abort();
  disableProgress(0, 0);
  fetch('/abort', { method: "GET" })
    .then(response => {
      response.text();
    })
    .then(response => {
      console.log("aborted.");
    })
    .catch(error => {
      console.log("abort error: " + error);
    });
}

function circleBoundary() {
  if(currentUI.value === "azure") {
    console.log("Please use the draw toolbar on the top left corner of the map to draw a circle boundary.");
    return;
  }
  // radius? construct a circle
  let radius = document.getElementById("radius").value;
  if (radius !== "") {
    //Clear any current boundaries before getting coordinates
    clearBoundaries();
    // make circle
    let centerCoords = currentMap.getCenter();

    // convert to m
    radius = Number(radius);

    // googleMap.addBoundary(new CircleBoundary(centerCoords, radius));
    currentMap.addBoundary(new CircleBoundary(centerCoords, radius));

    // googleMap.showBoundaries();

    bingMap.showBoundaries();
  }
}

function drawnBoundary() {
  if(currentUI.value === "azure") {
    console.log("Please use the draw toolbar on the top left corner of the map to draw a custom shape.");
    return;
  }
  console.log("using custom boundary polygon(s)");
  let boundaries = currentMap.retrieveDrawnBoundaries();
  for (let b of boundaries) {
    // googleMap.addBoundary(b);
    currentMap.addBoundary(b);
  }
}

function clearBoundaries() {
  currentMap.resetBoundaries();
}


function parseLatLngArray(a) {
  result = [];
  for (let p of a) {
    result.push({ lat: p[1], lng: p[0] });
  }
  return result;
}

function polyBounds(ps) {
  bounds = new google.maps.LatLngBounds();

  for (let p of ps) {
    bounds.extend(p);
  }
  return bounds;
}

function fillEngines() {
  $.ajax({
    url: "/getengines",
    success: function (result) {
      let html = "";
      //console.log(result);
      let es = JSON.parse(result);
      engines = {};
      //console.log(engines);
      for (let i = 0; i < es.length; i++) {
        html += "<input type='radio' id='" + es[i]['id']
        html += "' name='model' value='" + es[i]['id'] + "'"
        html += i == 0 ? " checked>" : ">"
        html += "<label for='" + es[i]['id'] + "'>" + es[i]['name'] + "</label><br>";
        engines[es[i]['id']] = es[i]['name'];
      }
      $("#engines").html(html);
    }
  });
}

function fillProviders() {
  // retrieve the backend providers
  $.ajax({
    url: "/getproviders",
    success: function (result) {
      let html = "";
      //console.log(result);
      let ps = JSON.parse(result);
      providers = {};
      //console.log(engines);
      for (let i = 0; i < ps.length; i++) {
        html += "<input type='radio' id='" + ps[i]['id']
        html += "_provider' name='provider' value='" + ps[i]['id'] + "_provider'"
        html += i == 0 ? " checked>" : ">"
        html += "<label for='" + ps[i]['id'] + "_provider'>" + ps[i]['name'] + "</label><br>";
        providers[ps[i]['id']] = ps[i]['name'];
      }
      $("#providers").html(html);

      // add change listeners for the backend provider radio box
      let rad = document.providers.provider;
      currentProvider = 'bing';//rad; //Bing Maps is the defacto provider for now

      // for (let r of rad) {
      //   r.addEventListener('change', function () {
      //     // no action right now
      //   });
      // }

      // and one for the file input box
      let fileBox = document.getElementById("upload_file");
      fileBox.addEventListener('change', () => {
        uploadImage();
      });

      // and one for the model upload box
      // let modelBox = document.getElementById("upload_model");
      // modelBox.addEventListener('change', () => {
      //   uploadModel();
      // });

      // and one for the dataset upload box
      // let datasetBox = document.getElementById("upload_dataset");
      // datasetBox.addEventListener('change', () => {
      //   uploadDataset();
      // });

    }
  });
}

function setMap(newMap) {
  if (currentUI !== null) {
    document.getElementById(currentUI.value + "Map").style.display = "none";
  }
  currentUI = newMap;
  handle = document.getElementById(currentUI.value + "Map");
  handle.style.display = "block";
  handle.style.width = "100%";
  handle.style.height = "100%";

  // let lastMap = currentMap;
  let zoom;
  let center;
  // if (lastmap) {
  //   zoom = currentMap.getZoom();
  //   center = currentMap.getCenter();
  // }

  if (currentUI.value === "upload") {
    document.getElementById("uploadsearchui").style.display = "block";
    document.getElementById("mapsearchui").style.display = "none";
    document.getElementById("fdetect").style.display = "none";
    document.getElementById("ftowers").style.display = "none";
    document.getElementById("fsave").style.display = "none";
    document.getElementById("freview").style.display = "none";
    // document.getElementById("ffilter").style.display = "none";
    document.getElementById("fadd").style.display = "none";
  } else if (currentUI.value === "google") {
    document.getElementById("uploadsearchui").style.display = "none";
    document.getElementById("mapsearchui").style.display = null;
    document.getElementById("fdetect").style.display = null;
    document.getElementById("ftowers").style.display = null;
    document.getElementById("fsave").style.display = null;
    document.getElementById("freview").style.display = null;
    // document.getElementById("ffilter").style.display = null;
    document.getElementById("fadd").style.display = null;
    currentMap = googleMap;
  } else if (currentUI.value === "bing") {
    document.getElementById("uploadsearchui").style.display = "none";
    document.getElementById("mapsearchui").style.display = null;
    document.getElementById("fdetect").style.display = null;
    document.getElementById("ftowers").style.display = null;
    document.getElementById("fsave").style.display = null;
    document.getElementById("freview").style.display = null;
    // document.getElementById("ffilter").style.display = null;
    document.getElementById("fadd").style.display = null;
    currentMap = bingMap;
    // recreate boundaries for bing
    let bs = bingMap.boundaries;
    bingMap.resetBoundaries();
    bs.map(b => bingMap.addBoundary(b));
    zoom = currentMap.getZoom();
    center = currentMap.getCenter();
  } else if (currentUI.value === "azure") {
    document.getElementById("uploadsearchui").style.display = "none";
    document.getElementById("mapsearchui").style.display = null;
    document.getElementById("fdetect").style.display = null;
    document.getElementById("ftowers").style.display = null;
    document.getElementById("fsave").style.display = null;
    document.getElementById("freview").style.display = null;
    // document.getElementById("ffilter").style.display = null;
    document.getElementById("fadd").style.display = null;
    currentMap = azureMap;
    // recreate boundaries for bing
    let bs = currentMap.boundaries;
    currentMap.resetBoundaries();
    bs.map(b => currentMap.addBoundary(b));
    zoom = currentMap.getZoom();
    center = currentMap.getCenter();
  }

  // set center and zoom

  if (typeof lastMap !== 'undefined') {
    if (currentMap.boundaries.length > 0) {
      currentMap.showBoundaries();
    }
    currentMap.setZoom(zoom);
    currentMap.setCenter(center);
  }

  // move all rectangles over to the new map
  Tile_tiles.forEach(t => t.update(currentMap));
  Detection_detections.forEach(d => d.update(currentMap))
}

function adjustConfidence() {
  Detection_minConfidence = confSlider.value / 100;
  for (let det of Detection_detections) {
    let meetsInside = reviewCheckBox.checked || det.inside;
    let meetsConf = det.conf >= Detection_minConfidence || det.p2 >= Detection_minConfidence;
    let meetsAddrConf = det.firstDet.maxConf >= Detection_minConfidence || det.firstDet.maxP2 >= Detection_minConfidence;
    det.firstDet.showAddr(meetsAddrConf && meetsInside);
    det.show(meetsConf && meetsInside);
    det.update();
  }
  document.getElementById('confpercent').innerText = confSlider.value;
}

function changeReviewMode() {
  if (reviewCheckBox.checked) {
    confSlider.value = 0;
  } else {
    confSlider.value = Math.round(DEFAULT_CONFIDENCE * 100);
  }
  adjustConfidence();
}

function augmentDetections() {
  Detection_detectionsAugmented = 0;
  //for (let det of Detection_detections) {
    for (let i = 0; i < Detection_detections.length; i++) {
      let det = Detection_detections[i];
    if (det.address !== "") {
      Detection_detectionsAugmented++;
      continue;
    }
    let loc = det.getCenterUrl();
     // call Bing maps api instead at:

     setTimeout((ix)=>{
      //console.log(ix+1);
      $.ajax({
      url: "https://dev.virtualearth.net/REST/v1/locationrecog/" + loc,
      data: {
        key: bak,
        includeEntityTypes: "address",
        output: "json",
      },
      success: function (result) {
        let addr = result['resourceSets'][0]['resources'][0]['addressOfLocation'][0]['formattedAddress'];
        det.augment(addr);
        afterAugment();
      }

    });
     },1000*i,i)
    // $.ajax({
    //   url: "https://maps.googleapis.com/maps/api/geocode/json",
    //   data: {
    //     latlng: loc,
    //     key: gak,
    //     location_type: "ROOFTOP",
    //     result_type: "street_address",
    //   },
    //   success: function (result) {
    //     let addr = "";
    //     if (result['status'] === "OK") {
    //       addr = result['results'][0]['formatted_address'];
    //       det.augment(addr);
    //       afterAugment();
    //     } else {
    //       addr = "(unable to determine address)";
    //       // console.log("Cannot parse address result for tower "+i+": "+JSON.stringify(result));

    //     }
    //     //det.augment(addr);
    //   }
    // });

  }

  //TODO-- Use Bing or Azure maps to Augment Detections
}

function afterAugment() {
  // wait for the last one
  if (Detection_detectionsAugmented !== Detection_detections.length) {
    return;
  }

  Detection.sort();
  Detection.generateList();

  // now hide low confidence values, sort the list and do the rest
  adjustConfidence();
}




function rad(x) {
  return x * Math.PI / 180;
};

// returns the Haversine distance between two points, in meters
function getDistance(p1, p2) {
  let R = 6378137; // Earth’s mean radius in meters
  let dLat = rad(p2[1] - p1[1]);
  let dLong = rad(p2[0] - p1[0]);
  let a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(rad(p1[1])) * Math.cos(rad(p2[1])) *
    Math.sin(dLong / 2) * Math.sin(dLong / 2);
  let c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  let d = R * c;
  return d;
};


function download(filename, data) {
  // create blob object with our data
  let blob = new Blob([data], { type: 'text/csv' });

  // create a  anchor element
  let elem = window.document.createElement('a');

  // direct it to the blob and filename
  elem.href = window.URL.createObjectURL(blob);
  elem.download = filename;

  // briefly insert it into the document, click it, remove it
  document.body.appendChild(elem);
  elem.click();
  document.body.removeChild(elem);
}



function download_dataset() {
  console.log("downloading dataset ...")
  include = []
  additions = []
  for (let det of Detection_detections) {
    if (det.idInTile !== -1 && det.conf >= Detection_minConfidence && det.selected) {
      include.push({ 'tile': det.tile, 'detection': det.idInTile, 'id': det.originalId });
      //console.log(" including detection #" + (det.originalId));
    }
    if (det.idInTile === -1) {
      tile = Tile_tiles[det.tile]
      additions.push({
        'tile': det.tile,
        'centerx': (((det.x1 + det.x2) / 2) - tile.x1) / (tile.x2 - tile.x1),
        'centery': (((det.y1 + det.y2) / 2) - tile.y1) / (tile.y2 - tile.y1),
        'w': (det.x2 - det.x1) / (tile.x2 - tile.x1),
        'h': (det.y1 - det.y2) / (tile.y1 - tile.y2)
      })
    }
  }

  // package all this up for the request
  let formData = new FormData();
  formData.append("include", JSON.stringify(include));
  formData.append("additions", JSON.stringify(additions));

  // post the arguments, get the dataset
  fetch("getdataset", { method: 'POST', body: formData })
    .then(response => response.blob())
    .then(blob => {
      // create a temp anchor element
      let elem = window.document.createElement('a');

      // direct it to the blob and filename
      elem.href = window.URL.createObjectURL(blob);
      elem.download = "dataset.zip";

      // briefly insert it into the document, click it, remove it
      document.body.appendChild(elem);
      elem.click();
      document.body.removeChild(elem);
    })
    .catch(error => {
      console.log("error in download: " + error);
    });
}

function download_csv() {
  text = "id,selected,inside_boundary,meets threshold,latitude (deg),longitude (deg),distance from center (m),address,confidence\n";
  for (let i = 0; i < Detection_detections.length; i++) {
    let det = Detection_detections[i];
    text += [
      i,
      det['selected'],
      reviewCheckBox.checked || det.inside,
      det['conf'] >= confSlider.value / 100,
      det.getCenter()[1].toFixed(8),
      det.getCenter()[0].toFixed(8),
      getDistance(det.getCenter(), currentMap.getCenter()).toFixed(1),
      ("\"" + det['address'] + "\""),
      det['conf'].toFixed(2)
    ].join(",") + "\n";
  }
  download("detections.csv", text);
}

function download_kml() {
  text = '<?xml version="1.0" encoding="UTF-8"?>\n';
  text += '<kml xmlns="http://www.opengis.net/kml/2.2">\n';
  text += "  <Document>\n";
  text += "<Style id='icon-1736-0F9D58-normal'><IconStyle><color>ffffa0a0</color><scale>1</scale>";
  text += "<Icon><href>https://maps.google.com/mapfiles/kml/pal4/icon35.png</href></Icon>";
  text += "</IconStyle><LabelStyle><scale>0</scale></LabelStyle></Style>\n";

  text += "<Style id='icon-1736-0F9D58-highlight'><IconStyle><color>ffa0a0ff</color><scale>1</scale>";
  text += "<Icon><href>https://maps.google.com/mapfiles/kml/pal4/icon35.png</href></Icon>";
  text += "</IconStyle><LabelStyle><scale>1</scale></LabelStyle></Style>\n";

  text += "<StyleMap id='icon-1736-0F9D58'><Pair><key>normal</key><styleUrl>";
  text += "#icon-1736-0F9D58-normal</styleUrl></Pair><Pair><key>highlight</key>";
  text += "<styleUrl>#icon-1736-0F9D58-highlight</styleUrl></Pair></StyleMap>\n\n";

  text += "<Style id='icon-1736-0F9D58-nodesc-normal'><IconStyle><color>ffffa0a0</color><scale>1</scale>";
  text += "<Icon><href>http://maps.google.com/mapfiles/kml/pal4/icon35.png</href></Icon>";
  text += "</IconStyle><LabelStyle><scale>0</scale></LabelStyle>";
  text += "<BalloonStyle><text><![CDATA[<h3>$[name]</h3>]]></text></BalloonStyle></Style>\n";

  text += "<Style id='icon-1736-0F9D58-nodesc-highlight'><IconStyle><color>ffa0a0ff</color><scale>1</scale>";
  text += "<Icon><href>http://maps.google.com/mapfiles/kml/pal4/icon35.png</href></Icon>";
  text += "</IconStyle><LabelStyle><scale>1</scale></LabelStyle>";
  text += "<BalloonStyle><text><![CDATA[<h3>$[name]</h3>]]></text></BalloonStyle></Style>\n";

  text += "<StyleMap id='icon-1736-0F9D58-nodesc'><Pair><key>normal</key><styleUrl>";
  text += "#icon-1736-0F9D58-nodesc-normal</styleUrl></Pair><Pair><key>highlight</key>";
  text += "<styleUrl>#icon-1736-0F9D58-nodesc-highlight</styleUrl></Pair></StyleMap>\n\n";

  for (let det of Detection_detections) {
    let inside = reviewCheckBox.checked || det.inside;
    if (det.conf >= Detection_minConfidence && det.selected && inside) {
      text += "    <Placemark>\n";
      text += '      <name>' + det.address + '</name>\n'
      text += '      <description>P(' + det.conf.toFixed(2) + ') at ' + det.address + ' ' + Tile_tiles[det.tile].metadata + '</description>\n';
      text += "      <styleUrl>#icon-1736-0F9D58</styleUrl>\n"
      text += '      <Point>\n';
      text += '        <altitudeMode>relativeToGround</altitudeMode>\n';
      text += '        <extrude>1</extrude>\n'
      text += '        <coordinates>' + det.getCenter()[0] + ',' + det.getCenter()[1] + ',300</coordinates>\n'
      text += '      </Point>\n';
      text += "    </Placemark>\n";
    }
  }
  text += "  </Document>\n";
  text += '</kml>\n';
  download("detections.kml", text);
}

//
// model upload functionality
//

function uploadModel() {
  let model = document.getElementById("upload_model").files[0];
  let formData = new FormData();

  Detection.resetAll();
  console.log("Model upload request in progress ...")

  formData.append("model", model);
  fetch('/uploadmodel', { method: "POST", body: formData })
    .then(response => {
      console.log("installed model " + model);
      fillEngines();
    })
    .catch(error => {
      console.log(error);
    });
}


//
// file upload functionality
//

function uploadImage() {
  let image = document.getElementById("upload_file").files[0];
  let engine = $('input[name=model]:checked', '#engines').val()
  let formData = new FormData();

  Detection.resetAll();
  console.log("Custom image detection request in progress ...")

  formData.append("image", image);
  formData.append("engine", engine)
  fetch('/getobjectscustom', { method: "POST", body: formData })
    .then(response => response.json())
    .then(response => {
      response = response[0];
      console.log(response.length + " object" + (response.length == 1 ? "" : "s") + " detected");
      console.log("loading file " + image.name);
      drawCustomImage("/uploads/" + image.name);
    })
    .catch(error => {
      console.log(error);
    });
}

function drawCustomImage(url) {
  let img = document.getElementById('canvas');
  img.src = url;
  if (img.complete) {
    removeCustomImage(url)
  } else {
    img.addEventListener('load', () => {removeCustomImage(url);}, {once:true});
  }
}

function removeCustomImage(url) {
  fetch('/rm'+url, { method: "GET"});
}




//
// upload dataset functionality
//

function uploadDataset() {
  if (Detection_detections.length > 0) {
    if (!window.confirm("This will erase current detections. Proceed?")) {
      return;
    }
  }

  let dataset = document.getElementById("upload_dataset").files[0];
  let formData = new FormData();

  Detection.resetAll();
  console.log("Dataset upload request in progress ...")
  let startTime = performance.now();


  formData.append("dataset", dataset);
  fetch('/uploaddataset', { method: "POST", body: formData })
    .then(response => response.json())
    .then(response => {
      processObjects(response, startTime);
    })
    .catch(error => {
      console.log(error);
    });
}

//
// estimate number of tiles
//

function estimateNumTiles(zoom, bounds) {
  // cop-out: do it from zoom, does not take window size into account
  let num = Math.pow(2, (19 - zoom) * 2 + 1);
  return Math.ceil(num);
}

//
// progress bar
//

let progressTimer = null;
let totalSecsEstimated = 0;
let secsElapsed = 0;
let numTiles = 0;
let secsPerTile = 0.25;
let dataPoints = 0;

function enableProgress(tiles) {
  document.getElementById("progress_div").style.display = "flex";

  progressTimer = setInterval(progressFunction, 100);
  numTiles = tiles;
  totalSecsEstimated = secsPerTile * numTiles;
  secsElapsed = 0;
}
function fatalError(msg) {
  document.getElementById("fatal_div").style.display = "flex";
  document.getElementById("fatal_div").innerHTML = "<center>" + msg + "</center>";
}

function disableProgress(time, actualTiles) {
  document.getElementById("progress_div").style.display = "none";

  clearInterval(progressTimer);
  if (time !== 0) {
    let secsPerTileLast = time / actualTiles;
    secsPerTile = (secsPerTile * dataPoints + secsPerTileLast) / (dataPoints + 1);
    dataPoints++;
  }
}
function progressFunction() {
  secsElapsed += 0.1;
  setProgress(secsElapsed / totalSecsEstimated * 100);
}

function setProgress(val) {
  document.getElementById("progress").value = String(val);
}


// debug helper: rerouting console.log into the window

class myConsole {
  constructor() {
    this.textArea = document.getElementById("output");
    // console.log("output area: " + this.textArea);
  }

  print(text) {
    this.textArea.innerText += text;
  }

  newLine() {
    this.textArea.innerHTML += "<br>";
    this.textArea.scrollTop = 99999;
  }

  log(text) {
    this.print(text);
    this.newLine();
  }
}

//
// initial position
//

function setMyLocation() {
  if (location.protocol === 'https:' && navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(showPosition);
  } else {
    currentMap.setCenter(nyc);
  }
}

function showPosition(position) {
  currentMap.setCenter([position.coords.longitude, position.coords.latitude]);
}


//
// zipcode lookup
//

function getZipcodePolygon(z) {
  if (z.startsWith("zipcode ")) {
    z = z.substring(8);
  } else if (z[0] === '"') {
    z = z.substring(1, 6);
  }
  fetch('/getzipcode?zipcode=' + z, { method: "GET" })
    .then(response => response.json())
    .then(response => {
      let polygons = parseZipcodeResult(response);
      if (polygons != []) {
        currentMap.resetBoundaries();
        for (let polygon of polygons) {
          currentMap.addBoundary(new PolygonBoundary(polygon[0]));
        }
        currentMap.showBoundaries();
      }
    })
    .catch(error => {
      console.log(error);
    });
}

function parseZipcodeResult(result) {
  if (result['type'] !== 'FeatureCollection') {
    return [];
  }

  let features = result['features'];
  let f = features[0];
  let geom = f['geometry']
  let coords = geom['coordinates'];
  return geom['type'] === 'Polygon' ? [coords]:coords;
}

// init actions
console = new myConsole();

// fillEngines();
fillProviders();
confSlider.value = Math.round(Detection_minConfidence * 100);

// Function to assign aria-labels to elements that need them
function assignAriaLabels() {
  // Select specific elements that typically need aria-labels
  const elementsNeedingAriaLabel = document.querySelectorAll('button, a, input, textarea, select, checkbox');

  // // Define a mapping of elements to their corresponding aria-labels
  // const ariaLabelMapping = {
  //     'submit-button': 'Submit Form',
  //     'cancel-button': 'Cancel Action',
  //     'username-input': 'Enter your username',
  //     'password-input': 'Enter your password',
  //     'email-input': 'Enter your email address',
  //     // Add more mappings as needed
  // };

  // Loop through each element
  elementsNeedingAriaLabel.forEach(element => {
      const id = element.id; // Get the element's ID
      if (!element.hasAttribute('aria-label')) {

              element.setAttribute('aria-label', element.id);
              // console.log(`Assigned aria-label to ${id}: ${element.id}`);

      }
  });
  const mapelementsNeedingAriaLabel=document.getElementById("bingMap").querySelectorAll('*');
   // Loop through each element
   mapelementsNeedingAriaLabel.forEach(element => {
    const id = element.id; // Get the element's ID
    if (!element.hasAttribute('aria-label')) {

            element.setAttribute('aria-label', element.id);
            // console.log(`Assigned aria-label to ${id}: ${element.id}`);

    }
});
}
function assignAriaLabelsToMapControl() {
  // // Example: Accessing zoom buttons
  // const zoomInButton = document.querySelector('.zoom-in-button-class'); // Use the correct class
  // const zoomOutButton = document.querySelector('.zoom-out-button-class'); // Use the correct class
  const rotateLeft = document.getElementById("bingMap").querySelector('#RotateLeftButton'); // Use the correct class
  const rotateRight = document.getElementById("bingMap").querySelector('#RotateRightButton'); // Use the correct class
  const rotate = document.getElementById("bingMap").querySelector('#RotateButton'); // Use the correct class
  const birdseye = document.getElementById("bingMap").querySelector('#BirdseyeV2ExitButton'); // Use the correct class
  const labelStyleSwitch = document.getElementById("bingMap").querySelector('.labelStyleSwitch'); // Use the correct class
  const labelTogglelabel = document.getElementById("bingMap").querySelector('.labelToggle_label'); // Use the correct class
  const chkToggle = document.getElementById("bingMap").querySelector('#navbarLabelToggleInput'); // Use the correct class
  const chkToggle1 = document.getElementById("bingMap").querySelector('#be2ToggleInput'); // Use the correct class
  if (labelStyleSwitch) {
    labelStyleSwitch.setAttribute('ID', 'lbllabelStyleSwitch');
  }

  if (chkToggle) {
    chkToggle.setAttribute('aria-label', 'lbllabelStyleSwitch');
  }
  if (labelTogglelabel) {
    labelTogglelabel.setAttribute('ID', 'lbllabelTogglelabel');
  }
  if (chkToggle1) {
    chkToggle1.setAttribute('aria-label', 'lbllabelStyleSwitch');
  }
  if (rotateLeft) {
    rotateLeft.setAttribute('aria-label', 'Rotate Left');
  }

  if (rotateRight) {
    rotateRight.setAttribute('aria-label', 'Rotate Right');
  }

  if (rotate) {
    rotate.setAttribute('aria-label', 'Rotate');
  }
  if (birdseye) {
    birdseye.setAttribute('aria-label', 'Birds Eye View');
  }
}

window.addEventListener('load', () => {
  assignAriaLabelsToMapControl();
});


console.log("TowerScout initialized.");
