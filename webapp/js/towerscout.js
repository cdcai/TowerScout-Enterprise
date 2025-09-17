
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

const upload = document.getElementById("upload_file");
const detectionsList = document.getElementById("checkBoxes");
const confSlider = document.getElementById("conf");
const reviewCheckBox = document.getElementById("review");
// dynamically adjust confidence of visible predictions
confSlider.oninput = adjustConfidence;
reviewCheckBox.onchange = changeReviewMode;
const DEFAULT_CONFIDENCE = 0.35;
let startTime = performance.now();
var formData;
var globaluser_id;
var globalrequest_id;
var CurrentSearcharea = "nonrural";


// Initialize and add the map
function initBingMap() {
  bingMap = new BingMap();
  currentMap = bingMap;

  // add change listeners for radio buttons

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
  ToggleTestEnvironment();
  getazmapTransactioncountjs(2);
  getClusterStatusjs();
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

function isRectangleInsidePolygon(x1, y1, x2, y2, polygon) {
  const corners = [
    { x: x1, y: y1 },
    { x: x1, y: y2 },
    { x: x2, y: y1 },
    { x: x2, y: y2 }
  ];

  return corners.every(corner => pointInPolygon(corner, polygon));
}
function pointInPolygon(point, polygon) {
  let x = point.x, y = point.y;
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    let xi = polygon[i].x, yi = polygon[i].y;
    let xj = polygon[j].x, yj = polygon[j].y;

    let intersect = ((yi > y) !== (yj > y)) &&
      (x < (xj - xi) * (y - yi) / (yj - yi + Number.EPSILON) + xi);
    if (intersect) inside = !inside;
  }

  return inside;
}

// function resultIntersectsPolygons(x1, y1, x2, y2, polygons) {
//   if (polygons.length === 0) {
//       return true;
//   }

//   const rect = [
//       [x1, y1], [x2, y1],
//       [x2, y2], [x1, y2],
//       [x1, y1]
//   ];

//   for (const poly of polygons) {
//       // Check edge intersections
//       for (let i = 0; i < 4; i++) {
//           const r1 = rect[i];
//           const r2 = rect[i + 1];
//           for (let j = 0; j < poly.length - 1; j++) {
//               const p1 = poly[j];
//               const p2 = poly[j + 1];
//               if (segmentsIntersect(r1, r2, p1, p2)) {
//                   return true;
//               }
//           }
//       }

//       // Check if rectangle is inside polygon
//       if (pointInPolygon(rect[0], poly)) {
//           return true;
//       }

//       // Check if polygon is inside rectangle
//       if (pointInPolygon(poly[0], rect)) {
//           return true;
//       }
//   }

//   return false;
// }
function check_bounds(x1, y1, x2, y2, bounds) {
  const [south, west, north, east] = bounds.map(parseFloat);
  return !(y1 < south || y2 > north || x2 < west || x1 > east);

}

/**
* This is a reusable function that sets the Azure Maps platform domain,
* signs the request, and makes use of any transformRequest set on the map.
* Use like this: `const data = await processRequest(url);`
*/
async function processRequest(url, map) {
  // Check if it's a template-style URL (e.g. fuzzy search)
  const isTemplate = url.includes('{azMapsDomain}');

  // Replace the domain placeholder if it's a template
  if (isTemplate) {
    url = url.replace('{azMapsDomain}', atlas.getDomain());
  }

  let requestParams = {
    url: url,
    headers: {}
  };

  // Only sign the request if it's a template (map SDK URL)
  if (isTemplate) {
    requestParams = map.authentication.signRequest({ url: url });
  }

  // Apply transformRequest if defined (for debugging or proxy use)
  const transform = map.getServiceOptions()?.transformRequest;
  if (typeof transform === 'function') {
    const transformed = transform(requestParams.url);
    if (transformed?.url) {
      requestParams = {
        url: transformed.url,
        headers: transformed.headers || {}
      };
    }
  }


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

function fetchGeometry(geometryId) {
  const url = `https://atlas.microsoft.com/search/polygon/json?api-version=1.0&geometries=${geometryId}&subscription-key=${azure_api_key}`;

  return fetch(url)
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      return data; // Return the geometry data
    })
    .catch(error => {
      console.error('Error fetching geometry:', error);
    });
}

class AzureMap extends TSMap {

  constructor() {
    super();
    this.map = new atlas.Map('azureMap', {
      center: [nyc[0], nyc[1]], // Reverse Bing
      zoom: 18,
      maxZoom: 20,
      disableStreetside: true,
      authOptions: {
        authType: 'subscriptionKey',
        subscriptionKey: azure_api_key
      },
      style: "road" // Ensure you're using vector tiles for clarity
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
    this.customLayers = [];

    // Listen for the map's ready event
    this.map.events.add('ready', () => {
      // Now it's safe to load drawing tools and add sources/layers
      this.loadDrawingTools();
      var drawingLayers = this.drawingManager.getLayers();
      if (drawingLayers !== undefined) {
        this.map.events.add("click", drawingLayers.polygonLayer, (e) => {
          // alert("Polygon clicked");
        });
      }

      let datasource = new atlas.source.DataSource('searchResultDataSource');
      this.map.sources.add(datasource);
      //Add a layer for rendering point data.
      // this.map.layers.add(new atlas.layer.SymbolLayer(datasource));

      //Create a jQuery autocomplete UI widget.
      var geocodeServiceUrlTemplate = 'https://{azMapsDomain}/search/fuzzy/json?typeahead=true&api-version=1.0&query={query}&language=en-US&lon={lon}&lat={lat}&countrySet=US&view=Auto';
      $("#azureSearch").autocomplete({
        minLength: 4,
        delay: 300,
        source: (request, response) => {
          const term = request.term.trim();
          const isZip = /^\d{5}(-\d{4})?$/.test(term); // ZIP or ZIP+4
          const latLonMatch = term.match(/^(-?\d+(\.\d+)?),\s*(-?\d+(\.\d+)?)$/); // Matches "lat, lon"
          const isLatLon = latLonMatch !== null;

          let requestUrl;

          if (isLatLon) {
            // // Handle reverse geocoding
            return;
          } else {
            // Use fuzzy search for all other inputs
            requestUrl = geocodeServiceUrlTemplate.replace('{query}', encodeURIComponent(term));

            if (isZip) {
              // Remove location bias for ZIP search
              requestUrl = requestUrl.replace('&lon={lon}', '').replace('&lat={lat}', '');
            } else {
              // Add map center bias
              const center = this.map.getCamera().center;
              requestUrl = requestUrl.replace('{lon}', center[0]).replace('{lat}', center[1]);
            }
          }

          processRequest(requestUrl, this.map).then(data => {
            const suggestions = [];

            const results = data.results || data.addresses || [];

            results.forEach(item => {
              const address = item.address || {};
              const postalCode = address.postalCode || '';
              const city = address.municipality || '';
              const state = address.countrySubdivision || '';
              const freeform = address.freeformAddress || `${item.position.lat}, ${item.position.lon}`;
              const poiName = item.poi?.name;

              let label = freeform;
              if (postalCode && city && state) {
                label = `${postalCode} - ${city}, ${state}`;
              }
              if (poiName) {
                label = `${poiName} (${label})`;
              }

              suggestions.push({
                ...item,
                label: label,
                value: label
              });
            });

            response(suggestions);
          });
        },
        select: (event, ui) => {
          event.preventDefault();
          document.getElementById("azureSearch").value = ui.item.address.freeformAddress
          //Remove any previous added data from the map.
          datasource.clear();

          const geometryId = ui.item?.dataSources?.geometry?.id;
          if (geometryId) {
            fetchGeometry(geometryId).then(geometryData => {
              const polys = [];

              const extractPoints = (coords) => {
                return coords.map(coord => [coord[0], coord[1]]);
              };

              if (geometryData) {
                let coordinates = geometryData.additionalData[0].geometryData.features[0].geometry.coordinates;
                if (geometryData.additionalData[0].geometryData.features[0].geometry.type === "MultiPolygon") { // This means it is a multi polygon...
                  coordinates.map(polygon => {
                    const muliPolygonPolygon = new atlas.data.Polygon(polygon);
                    datasource.add(new atlas.data.Feature(muliPolygonPolygon));
                    const points = extractPoints(muliPolygonPolygon.coordinates[0]);
                    polys.push(new PolygonBoundary(points));
                  })
                }

                if (geometryData.additionalData[0].geometryData.features[0].geometry.type === "Polygon") {
                  const polygon = new atlas.data.Polygon(coordinates);
                  datasource.add(new atlas.data.Feature(polygon));
                  const points = extractPoints(polygon.coordinates[0]);
                  polys.push(new PolygonBoundary(points));
                }


                const polygonStyle = {
                  fillColor: 'rgba(0, 0, 255, 0.5)',
                  strokeColor: 'blue',
                  strokeWidth: 1
                };

                const polygonLayer = new atlas.layer.PolygonLayer(datasource, 'searchResultPolygon', {
                  fillColor: polygonStyle.fillColor,
                  strokeColor: polygonStyle.strokeColor,
                  strokeWidth: polygonStyle.strokeWidth
                });
                this.map.layers.add(polygonLayer);

                this.boundaries = polys;
              }
            });
            //Zoom the map into the selected location.
            this.map.setCamera({
              bounds: [
                ui.item.viewport.topLeftPoint.lon, ui.item.viewport.btmRightPoint.lat,
                ui.item.viewport.btmRightPoint.lon, ui.item.viewport.topLeftPoint.lat
              ],
              padding: 0
            });
          }
          // 🔍 Check for fuzzy result (with viewport) or reverse (with position)
          if (ui.item?.viewport) {
            this.map.setCamera({
              bounds: [
                ui.item.viewport.topLeftPoint.lon, ui.item.viewport.btmRightPoint.lat,
                ui.item.viewport.btmRightPoint.lon, ui.item.viewport.topLeftPoint.lat
              ],
              padding: 0
            });
          } else if (ui.item?.position) {
            // Reverse geocode result — center map using lat/lon
            let lat, lon;

            if (typeof ui.item.position === 'string') {
              // Position is like "40.712967,-74.007301"
              [lat, lon] = ui.item.position.split(',').map(parseFloat);
            } else {
              // Just in case Azure ever returns an object
              lat = ui.item.position.lat;
              lon = ui.item.position.lon;
            }

            this.map.setCamera({
              center: [lon, lat],
              zoom: 14
            });
          }

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
      $("#azureSearch").on("keydown", (event) => { // Use arrow function here
        if (event.key === "Enter") {
          const term = $("#azureSearch").val().trim(); // Directly reference the input element
          const latLonMatch = term.match(/^(-?\d+(\.\d+)?),\s*(-?\d+(\.\d+)?)$/); // Matches "lat, lon"

          if (latLonMatch) {
            // Handle reverse geocoding
            const lat = latLonMatch[1];
            const lon = latLonMatch[3];
            const requestUrl = `https://atlas.microsoft.com/search/address/reverse/json?api-version=1.0&query=${lat},${lon}&subscription-key=${azure_api_key}`;

            // Call your processRequest function to fetch the result
            processRequest(requestUrl, this.map).then(data => {
              const results = data.results || data.addresses || [];

              if (results.length > 0) {
                const uiItem = {
                  address: {
                    freeformAddress: results[0].address.freeformAddress // Set the freeform address
                  },
                  position: {
                    lat: results[0].position.lat,
                    lon: results[0].position.lon
                  },
                  viewport: results[0].viewport // Assuming you want to use the viewport as well
                };

                // Remove any previous added data from the map.
                datasource.clear();

                // Handle geometry if available
                const geometryId = results[0]?.dataSources?.geometry?.id; // Adjust as necessary
                if (geometryId) {
                  fetchGeometry(geometryId).then(geometryData => {
                    const polys = [];

                    const extractPoints = (coords) => {
                      return coords.map(coord => [coord[0], coord[1]]);
                    };

                    if (geometryData) {
                      let coordinates = geometryData.additionalData[0].geometryData.features[0].geometry.coordinates;
                      if (geometryData.additionalData[0].geometryData.features[0].geometry.type === "MultiPolygon") {
                        coordinates.map(polygon => {
                          const multiPolygon = new atlas.data.Polygon(polygon);
                          datasource.add(new atlas.data.Feature(multiPolygon));
                          const points = extractPoints(multiPolygon.coordinates[0]);
                          polys.push(new PolygonBoundary(points));
                        });
                      } else if (geometryData.additionalData[0].geometryData.features[0].geometry.type === "Polygon") {
                        const polygon = new atlas.data.Polygon(coordinates);
                        datasource.add(new atlas.data.Feature(polygon));
                        const points = extractPoints(polygon.coordinates[0]);
                        polys.push(new PolygonBoundary(points));
                      }

                      const polygonStyle = {
                        fillColor: 'rgba(0, 0, 255, 0.5)',
                        strokeColor: 'blue',
                        strokeWidth: 1
                      };

                      const polygonLayer = new atlas.layer.PolygonLayer(datasource, 'searchResultPolygon', {
                        fillColor: polygonStyle.fillColor,
                        strokeColor: polygonStyle.strokeColor,
                        strokeWidth: polygonStyle.strokeWidth
                      });
                      this.map.layers.add(polygonLayer);

                      this.boundaries = polys;
                    }
                  });
                }

                // Zoom the map into the selected location
                if (uiItem.viewport) {
                  this.map.setCamera({
                    bounds: [
                      uiItem.viewport.topLeftPoint.lon, uiItem.viewport.btmRightPoint.lat,
                      uiItem.viewport.btmRightPoint.lon, uiItem.viewport.topLeftPoint.lat
                    ],
                    padding: 0
                  });
                } else if (uiItem.position) {
                  // Center map using lat/lon
                  this.map.setCamera({
                    bounds: [
                      lon, lat,
                      lon, lat
                    ],
                    padding: 0
                  });
                }
              } else {
                console.log("No results found.");
              }
            });
          } else {
            // console.log("Invalid input. Please enter a valid address or lat, lon.");
          }
        }
      });
      let currentStyle = this.map.getStyle();
      this.map.events.add('styledata', () => {
        const newStyle = this.map.getStyle();
        if (currentStyle.style !== newStyle.style) {
          console.log(`Switched from ${currentStyle.style} to ${newStyle.style}`);

          if ((currentStyle.style === 'road' && newStyle.style === 'satellite') || (currentStyle.style === 'satellite' && newStyle.style === 'road')) {
            console.log('Detected switch from Road to Satellite or Satellite to Road!');
            const datasource = this.drawingManager.getSource();
            if (!this.map.sources.getById(datasource.getId())) {
              this.map.sources.add(datasource);
            }

          }

          currentStyle = newStyle;
        }
      })

      this.map.getCanvasContainer().addEventListener('contextmenu', function (e) {
        e.preventDefault();
      });
      this.map.events.add('contextmenu', (e) => {
        // const geoPosition = this.map(e.position);
        const contextMenu = document.getElementById('mapContextMenu');

        // Store coordinates on the menu element for later use
        contextMenu.dataset.lat = e.position[1];
        contextMenu.dataset.lng = e.position[0];
        const coordsText = `${e.position[1]}, ${e.position[0]}`;
        // Display coordinates in the div
        document.getElementById('coordText').innerText = `Copy coordinates lat,lng: ${coordsText}`;
        // Use the original event for screen coordinates
        const pageX = e.originalEvent.pageX;
        const pageY = e.originalEvent.pageY;

        contextMenu.style.left = `${pageX}px`;
        contextMenu.style.top = `${pageY}px`;
        contextMenu.style.display = 'block';
      });
      this.map.events.add('click', () => {
        document.getElementById('mapContextMenu').style.display = 'none';
      });



      this.map.setUserInteraction({
        dragPanInteraction: true,
        mouseWheelZoomInteraction: true,
        doubleClickZoomInteraction: true,
        touchZoomInteraction: true
      });


    });
  }
  reinitDrawingManager() {
    const source = this.drawingManager.getSource();

    // If layers were removed, this forces reinitialization
    this.map.sources.remove(source);
    this.map.sources.add(source);

    // Optional: re-add shape layers if you have custom styling
  }
  loadDrawingTools() {
    // Load the DrawingTools module
    this.drawingManager = new atlas.drawing.DrawingManager(this.map,
      {
        mode: null,
        toolbar: new atlas.control.DrawingToolbar({
          position: 'top-left',
          style: 'light',
          buttons: ['draw-point', 'draw-polygon', 'draw-line', 'draw-circle', 'draw-rectangle', 'edit-geometry', 'erase-geometry'] // Include only the desired tools
        }),

      });
    var layers = this.drawingManager.getLayers();
    layers.lineLayer.setOptions({
      strokeColor: 'blue',
      strokeWidth: 1
    });
    layers.polygonOutlineLayer.setOptions({
      strokeColor: 'blue'
    });
    layers.pointLayer.setOptions({
      iconOptions: {
        image: 'pin-round-darkblue',  // Try other built-in Azure Maps icons
        anchor: 'center',
        size: 0.5,
        allowOverlap: true
      }
    });

    this.map.events.add('drawingchanging', this.drawingManager, (shape) => { this.measureShape(shape) });
    this.map.events.add('drawingchanged', this.drawingManager, (shape) => { this.measureShape(shape) });
    this.map.events.add('drawingcomplete', this.drawingManager, () => { this.getDrawnBoundariesShapes() });
  }


  getDrawnBoundariesShapes() {
    // There are no detections - user is drawing a boundary to search
    if (!detectionsList.hasChildNodes() && Detection_detections.length == 0) {
      let boundaries = currentMap.retrieveDrawnBoundaries();
      for (let b of boundaries) {
        currentMap.addBoundary(b);
      }
    }
    else {
      // this.retrieveDrawnShapes();
    }
  }
  retrieveDrawnBoundaries() {
    // 1. Get shapes from Drawing Manager
    let shapes = this.drawingManager.getSource().shapes;

    // 2. Prepare a new DataSource for map display
    const existingCustomDataSource = currentMap.map.sources.getById('customDataSource');
    var customDataSource = existingCustomDataSource ?? new atlas.source.DataSource('customDataSource');
    if (!existingCustomDataSource) {
      this.map.sources.add(customDataSource);
    }

    let polygonFeature, coordinates, points, poly;
    const polys = [];
    // 3. Loop through shapes and extract circle geometry
    shapes.forEach(shape => {
      if (shape.properties?.shape === 'Circle' || shape.circlePolygon) {
        polygonFeature = new atlas.data.Feature(
          new atlas.data.Polygon(shape.circlePolygon.geometry.coordinates)
        );
        // currentMap.boundaries.push(new PolygonBoundary(shape.circlePolygon.geometry.coordinates[0]));
        poly = new PolygonBoundary(shape.circlePolygon.geometry.coordinates[0]);
        polys.push(poly);
      }
      else if (shape.data.geometry.type == "Point") {


      }
      else if (shape.data.geometry.type == "LineString") {
        coordinates = shape.data.geometry.coordinates;
        const isClosed = coordinates.length > 2 &&
          coordinates[0][0] === coordinates[coordinates.length - 1][0] &&
          coordinates[0][1] === coordinates[coordinates.length - 1][1];

        if (isClosed) {
          polygonFeature = new atlas.data.Feature(
            new atlas.data.Polygon(coordinates)
          );
          // Push the boundaries
          points = coordinates.map(coord => [coord[0], coord[1]]);
          poly = new PolygonBoundary(points);

        }

      }
      else {
        coordinates = shape.data.geometry.coordinates[0];
        // Create a polygon feature
        polygonFeature = new atlas.data.Feature(new atlas.data.Polygon(coordinates));
        // Push the boundaries
        points = coordinates.map(coord => [coord[0], coord[1]]);
        poly = new PolygonBoundary(points);

      }

      // Add to map's DataSource
      if (polygonFeature != undefined) {
        customDataSource.add(polygonFeature);
        // Push boundaries to current map
        polys.push(poly);

        // 4. Add a PolygonLayer to display the circle outline

        this.map.layers.add(new atlas.layer.LineLayer(customDataSource, null, {
          strokeColor: 'blue',
          strokeWidth: 2
        }));

      }
      // Remove from DrawingManager
      if (shape.data.geometry.type !== "Point") {
        if (this.drawingManager.getSource().getShapes().includes(shape)) {
          this.drawingManager.getSource().remove(shape);
        }
        if (customDataSource.getShapes().includes(shape)) {
          customDataSource.remove(shape);
        }

      }
    });
    console.log(`Polys: ${polys}`);
    this.boundaries = polys;
    this.showBoundaries()
    return polys;
  }
  retrieveDrawnShapes() {
    // 1. Get shapes from Drawing Manager
    let shapes = this.drawingManager.getSource().shapes;

    // 2. Prepare a new DataSource for map display

    let customDataSource = new atlas.source.DataSource();
    this.map.sources.add(customDataSource);
    let polygonFeature;
    let coordinates;
    // 3. Loop through shapes and extract circle geometry
    shapes.forEach(shape => {
      if (shape.properties?.shape === 'Circle' || shape.circlePolygon) {
        polygonFeature = new atlas.data.Feature(
          new atlas.data.Polygon(shape.circlePolygon.geometry.coordinates)
        );
        currentMap.boundaries.push(new PolygonBoundary(shape.circlePolygon.geometry.coordinates[0]));
      }
      else if (shape.data.geometry.type == "Point") {
        coordinates = shape.data.geometry.coordinates;

        const lon = coordinates[0];
        const lat = coordinates[1];

        const offset = 0.000025; // approx ~111m (adjust based on zoom level and context)

        // Define square polygon around the point
        const squareCoords = [
          [lon - offset, lat - offset],
          [lon - offset, lat + offset],
          [lon + offset, lat + offset],
          [lon + offset, lat - offset],
          [lon - offset, lat - offset] // close the polygon
        ];

        // Create a polygon feature
        polygonFeature = new atlas.data.Feature(new atlas.data.Polygon(squareCoords));


      }
      else if (shape.data.geometry.type == "LineString") {
        coordinates = shape.data.geometry.coordinates;
        const isClosed = coordinates.length > 2 &&
          coordinates[0][0] === coordinates[coordinates.length - 1][0] &&
          coordinates[0][1] === coordinates[coordinates.length - 1][1];

        if (isClosed) {
          polygonFeature = new atlas.data.Feature(
            new atlas.data.Polygon(coordinates)
          );
          const points = coordinates.map(coord => [coord[0], coord[1]]);
          const poly = new PolygonBoundary(points);
          currentMap.addBoundary(poly);
        }

      }
      else {
        coordinates = shape.data.geometry.coordinates[0];
        // Create a polygon feature
        polygonFeature = new atlas.data.Feature(new atlas.data.Polygon(coordinates));
        const points = coordinates.map(coord => [coord[0], coord[1]]);
        const poly = new PolygonBoundary(points);
        currentMap.addBoundary(poly);
      }

      // Add to map's DataSource
      if (polygonFeature != undefined) {
        customDataSource.add(polygonFeature);


        // 4. Add a PolygonLayer to display the circle outline

        this.map.layers.add(new atlas.layer.LineLayer(customDataSource, null, {
          strokeColor: 'blue',
          strokeWidth: 2
        }));
      }
      // Remove from DrawingManager
      if (shape.data.geometry.type !== "Point") {
        this.drawingManager.getSource().remove(shape);
      }


    });

  }

  clearShapes() {

    this.drawingManager?.getSource()?.clear();
    this.map.sources.getById('circleDataSource')?.clear();
    this.removeLayerById("circleShapeLayer");
    this.clearAllCustomLayers();
    this.drawingManager?.setOptions({ mode: 'idle' });;  // Reset drawing mode

  }

  clearCustomBoundaryShapes() {
    const source = this.drawingManager.getSource();
    const shapes = source.getShapes();

    if (shapes && shapes.length > 0) {
      source.clear();

    }
    this.drawingManager?.getSource()?.clear();

  }

  clearAll() {
    Detection.resetAll();
    this.clearShapes();
    this.resetBoundaries();
    this.clearAllCustomLayers();
    this.clearCustomDataSource();
    ToggleSearchArea('nonrural');
  }
  hideAllDataSources() {
    var layers = this.map.layers.getLayers();  // Get all layers on the map

    layers.forEach(function (layer) {
      // Check if the layer is associated with any DataSource

      layer.setOptions({ visible: false });  // Hide the layer

    });
    // console.log('All DataSource layers have been hidden');
  }
  // Function to clear all layers from the map
  clearAllCustomLayers() {
    if (this.customLayers.length > 0) { // Check if there are any layers to remove
      this.customLayers.forEach(layer => {
        currentMap.map.layers.remove(layer); // Remove each layer from the map
      });
      this.customLayers = []; // Clear the array after removal

    }
  }
  clearCustomDataSource() {
    this.map.sources.getById('customDataSource')?.clear();
  }
  // Function to remove a layer by its id
  removeLayerById(layerId) {
    var layers = this.map.layers.getLayers(); // Get all layers in the map
    layers.forEach(function (layer) {
      if (layer.id === layerId) {
        currentMap.map.layers.remove(layer); // Remove the layer with the matching id
      }
    });
  }
  getBounds() {
    return this.map.getCamera().bounds;
  }

  getCenter() {
    let b = this.getBounds();
    return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
  }

  fitBounds(b) {
    this.map.setCamera({
      bounds: [
        [b[0], b[1]], // Southwest
        [b[2], b[3]]  // Northeast
      ],
      padding: 0,
      zoom: 18
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
    return;
  }
  showBoundaries() {
    // Set map bounds to fit the union of all active boundaries
    if (this.boundaries.length > 0) {
      var polygon = this.boundaries[0].points;
      var minX = Math.min(...polygon.map(p => p[0]));
      var minY = Math.min(...polygon.map(p => p[1]));
      var maxX = Math.max(...polygon.map(p => p[0]));
      var maxY = Math.max(...polygon.map(p => p[1]));

      var bounds = atlas.data.BoundingBox.fromEdges(minX, minY, maxX, maxY);
      this.map.setCamera({ bounds, padding: 30 });
    }
  }

  resetBoundaries() {
    this.boundaries = [];
    this.drawingManager?.getSource()?.clear();
    this.map.sources.getById('circleDataSource')?.clear();
    this.map.sources.getById('searchResultDataSource')?.clear();
  }

  hasShapes() {
    const shapes = this.drawingManager.getSource().shapes;
    return shapes && shapes.length > 0;
  }

  addShapes() {
    let shapes = this.drawingManager.getSource().shapes;
    let x1;
    let y1;
    let x2;
    let y2;
    let bounds;
    if (shapes && shapes.length > 0) {
      console.log('Retrieved ' + shapes.length + ' from the drawing manager.');
      for (let s of shapes) {
        if (s.data.geometry.type === "Point") {
          const [lon, lat] = s.data.geometry.coordinates;

          const offset = 0.000025; // Adjust for square size (~11m at equator)

          // Create corners of square around the point
          const squareCoords = [
            [lon - offset, lat - offset],
            [lon - offset, lat + offset],
            [lon + offset, lat + offset],
            [lon + offset, lat - offset],
            [lon - offset, lat - offset] // close polygon
          ];
          // Now create a Polygon shape from this square
          const squarePolygon = new atlas.data.Feature(new atlas.data.Polygon([squareCoords]));
          // Get bounds from the polygon feature
          bounds = atlas.data.BoundingBox.fromData(squarePolygon);
          // Get bounding box corners
          x1 = bounds[0]; // West (min lon)
          y1 = bounds[3]; // North (max lat)
          x2 = bounds[2]; // East (max lon)
          y2 = bounds[1]; // South (min lat)
          console.log(`x1: ${x1}, y1: ${y1}, x2: ${x2}, y2: ${y2}`);
          let PointDataSource = new atlas.source.DataSource();
          // Optional: draw it on the map
          PointDataSource.add(squarePolygon);
          this.map.sources.add(PointDataSource);
        }
        else {
          console.log("Adding " + s.getBounds().toString());

          x1 = s.getBounds()[0];
          y1 = s.getBounds()[3];
          x2 = s.getBounds()[2];
          y2 = s.getBounds()[1];
        }
        // If a circle is drawn, draw a square using the center and radius of the circle
        if (s.properties?.shape === 'Circle' || s.circlePolygon) {
          x1 = s.getBounds()[0];
          y1 = s.getBounds()[1];
          x2 = s.getBounds()[2];
          y2 = s.getBounds()[3];
          const centerLng = (x1 + x2) / 2;
          const centerLat = (y1 + y2) / 2;
          const center = [centerLng, centerLat];

          // 3. Estimate radius in meters using getDistanceTo (from center to one edge)
          const pointOnEdge = [x2, centerLat]; // East edge
          const radius = atlas.math.getDistanceTo(center, pointOnEdge); // in meters

          // 4. Use getDestination to find the square corners
          const north = atlas.math.getDestination(center, radius, 0);    // bearing 0° (North)
          const east = atlas.math.getDestination(center, radius, 90);    // 90° (East)
          const south = atlas.math.getDestination(center, radius, 180);  // 180° (South)
          const west = atlas.math.getDestination(center, radius, 270);   // 270° (West)

          // 5. Construct square using those points
          const squareCoords = [[
            [west[0], north[1]],  // Top-left
            [east[0], north[1]],  // Top-right
            [east[0], south[1]],  // Bottom-right
            [west[0], south[1]],  // Bottom-left
            [west[0], north[1]]   // Closing loop
          ]];

          // Step 4: Create a polygon and add to map
          const circleSquarePolygon = new atlas.data.Feature(new atlas.data.Polygon(squareCoords));


          let CircleSquareDataSource = new atlas.source.DataSource();
          CircleSquareDataSource.add(circleSquarePolygon);
          this.map.sources.add(CircleSquareDataSource);
          x1 = s.getBounds()[0];
          y1 = s.getBounds()[3];
          x2 = s.getBounds()[2];
          y2 = s.getBounds()[1];

        }


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
            'ct', 1.0, tileId, -1 /*id_in_tile*/, true, true, 1.0, 0);

          tile = Tile_tiles[tileId];

          det["silverx1"] = (det["x1"] - (tile["lng"] - 0.5 * tile["w"])) / tile["w"];
          det["silverx2"] = (det["x2"] - (tile["lng"] - 0.5 * tile["w"])) / tile["w"];

          det["silvery1"] = (tile["lat"] + 0.5 * tile["h"] - det["y1"]) / tile["h"];
          det["silvery2"] = (tile["lat"] + 0.5 * tile["h"] - det["y2"]) / tile["h"];
          // }


          det["image_hash"] = tile["image_hash"];
          det["uuid"] = tile["uuid"];
          // det.update();
        }

      }
      this.clearCustomBoundaryShapes();
      augmentDetections(true);


    } else {
      console.log('No shapes in the drawing manager.');
    }
  }

  getBoundariesStr() {
    let result = [];
    for (let b of this.boundaries) {
      result.push(b.toString())
    }
    return "[" + result.join(",") + "]";
  }

  getZoom() {
    return this.map.getCamera().zoom;
  }

  getBoundsUrl() {
    const bounds = this.map.getCamera().bounds;
    return [bounds[1], bounds[0], bounds[3], bounds[2]];
  }

  measureShape(shape) {
    var msg = '';

    if (shape.isCircle()) {
      document.getElementById("radius").value = shape.getProperties().radius;
    }

  }

  //Draw bounding boxes

  makeMapRect(o, listener) {
    try {
      let locs = [
        new atlas.data.Point(o.y1, o.x1),
        new atlas.data.Point(o.y1, o.x2),
        new atlas.data.Point(o.y2, o.x2),
        new atlas.data.Point(o.y2, o.x1),
        new atlas.data.Point(o.y1, o.x1)
      ];
      // Create the four corners of the polygon
      const boundingBoxCoordinates = [
        [o.x1, o.y1],  // Top-left corner
        [o.x2, o.y1],  // Top-right corner
        [o.x2, o.y2],  // Bottom-right corner
        [o.x1, o.y2],  // Bottom-left corner
        [o.x1, o.y1]   // Closing the polygon (back to top-left)
      ];


      // });
      // Create a polygon (bounding box) from the coordinates
      var boundingBoxPolygon = new atlas.data.Polygon([boundingBoxCoordinates]);
      // Create a data source and add the bounding box (polygon) to it


      //Create dummy layer to avoid the obj ref error due to the click event handler
      var fillLayer;

      if (o.classname == "tile") {

        var tiledataSource = new atlas.source.DataSource(null);
        tiledataSource.add(boundingBoxPolygon);
        currentMap.map.sources.add(tiledataSource);
        o.dataSourceID = tiledataSource.id;
        var tileborderLayer = new atlas.layer.LineLayer(tiledataSource, null, {
          strokeColor: 'blue',
          strokeWidth: 1                 // Border width
        });
        currentMap.map.layers.add(tileborderLayer);
        this.customLayers.push(tileborderLayer);
        o.borderLayerID = tileborderLayer.id;

        fillLayer = new atlas.layer.PolygonLayer(tiledataSource, null, {
          fillColor: 'rgba(0, 0, 0, 0)',      // No fill color (transparent)
        });
        currentMap.map.layers.add(fillLayer);
        this.customLayers.push(fillLayer);
        o.fillLayerID = fillLayer.id;

        this.map.events.add('contextmenu', fillLayer, (e) => {
          const contextMenu = document.getElementById('mapContextMenu');

          // Convert pixel position to geographic coordinates
          const position = e.position; // or pixelToPosition
          const lat = position[1];
          const lng = position[0];

          // Store coords in dataset for copy action
          contextMenu.dataset.lat = lat;
          contextMenu.dataset.lng = lng;

          // Show readable coordinates in the div
          document.getElementById('coordText').innerText = `Copy coordinates (lat,lng): ${lat}, ${lng}`;

          // Show context menu at mouse position
          const pageX = e.originalEvent.pageX;
          const pageY = e.originalEvent.pageY;
          contextMenu.style.left = `${pageX}px`;
          contextMenu.style.top = `${pageY}px`;
          contextMenu.style.display = 'block';
        });
      }
      else {
        var dataSource = new atlas.source.DataSource(null);
        currentMap.map.sources.add(dataSource);
        dataSource.add(boundingBoxPolygon);
        o.dataSourceID = dataSource.id;
        // console.log("dataSourceID:" + o.dataSourceID);
        // Create a layer for just the border of the bounding box/tile (without fill)
        var borderLayer = new atlas.layer.LineLayer(dataSource, null, {
          strokeColor: 'red',
          strokeWidth: 1                // Border width
        });
        currentMap.map.layers.add(borderLayer);
        this.customLayers.push(borderLayer);
        o.borderLayerID = borderLayer.id;
        fillLayer = new atlas.layer.PolygonLayer(dataSource, null, {
          strokeColor: 'red',
          fillColor: 'rgba(255, 0, 0, 0.3)', // Semi-transparent red fill
          strokeWidth: 1                  // Border width
        });
        currentMap.map.layers.add(fillLayer, null);
        this.customLayers.push(fillLayer);
        o.fillLayerID = fillLayer.id;;
        this.map.events.add('click', fillLayer, function (e) {
          // console.log('Polygon clicked:', e);
        });
      }


      return boundingBoxPolygon;

    }
    catch (error) {
      console.log('An error occurred makeMapRect: ' + error);  // This won't execute
    }
    finally {

    }
  }

  colorMapRect(o, color) {
    try {
      // Create a fill color with opacity

      let fcolor = Microsoft.Maps.Color.fromHex(color);
      let alpha = o.opacity;
      // // Convert to RGB string format
      let fillcolor = this.colorToRGBA(fcolor, alpha);
      let FilllayertoHighlight = currentMap.getLayerById(o.fillLayerID);
      FilllayertoHighlight.setOptions({
        strokeColor: color,
        fillColor: fillcolor, // Semi-transparent fill
        strokeWidth: 1
      });

    }
    catch (error) {
      console.log('An error occurred colorMapRect: ' + error);  // This won't execute
    }
    finally {

    }
  }
  // Convert Microsoft.Maps.Color to an RGBA string
  colorToRGBA(color, alpha) {
    return 'rgba(' + color.r + ', ' + color.g + ', ' + color.b + ', ' + alpha + ')';
  }

  updateMapRect(o, onoff) {
    try {
      if (onoff) {
        o.opacity = 1
        // console.log("o.fillLayerID:" + o.fillLayerID);
        let FilllayertoHide = currentMap.getLayerById(o.fillLayerID);
        FilllayertoHide.setOptions({
          visible: true
        });
        let BorderLayertoHide = currentMap.getLayerById(o.borderLayerID);
        BorderLayertoHide.setOptions({
          visible: true
        });
        // console.log("o.dataSourceID:" + o.dataSourceID);
        this.map.sources.getById(o.dataSourceID)?.setOptions({
          visible: true
        });
        o.visibilitySetto = true
      }
      else {
        // console.log("o.fillLayerID:" + o.fillLayerID);
        let FilllayertoHide = currentMap.getLayerById(o.fillLayerID);
        FilllayertoHide.setOptions({
          visible: false
        });
        let BorderLayertoHide = currentMap.getLayerById(o.borderLayerID);
        BorderLayertoHide.setOptions({
          visible: false
        });
        // console.log("o.dataSourceID:" + o.dataSourceID);
        this.map.sources.getById(o.dataSourceID)?.setOptions({
          visible: false
        });
        o.visibilitySetto = false
      }

    }
    catch (error) {
      console.log('An error occurred updateMapRect: ' + error);  // This won't execute
    }
    finally {

    }
  }
  // Function to get a layer by its id
  getLayerById(layerId) {
    var layers = this.map.layers.getLayers(); // Get all layers in the map
    return layers.find(function (layer) {
      return layer.id === layerId; // Compare the id of each layer
    });
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
      manager.attachAutosuggest('#bingSearch', '#bingSearchBoxContainer', selectedSuggestion);
    });

    document.getElementById('bingSearch').addEventListener('change', Search);

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
        where: document.getElementById('bingSearch').value,
        callback: getBoundary,
        errorCallback: function (e) {
          //If there is an error, alert the user about it.
          //alert("No results found.");
        }
      };
      searchManager.geocode(geocodeRequest);
    }
    function getBoundary(geocodeResult) {
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
              console.log(`Could not find boundary for ${document.getElementById('bingSearch').value}`)
            }
          }
        );
      }
    }

    function selectedSuggestion(result) {
      bingMap.entities.clear;
      var geocodeRequest;
      if ((result.entitySubType == "Address") || (result.entityType == "PostalAddress")) {
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
      else {
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

    for (var i = 0; i < this.map.entities.getLength(); i++) {
      var entity = this.map.entities.get(i);
      if (entity instanceof Microsoft.Maps.Polygon) {
        var vertices = entity.getLocations();
        locations = locations.concat(vertices);
      }
    }

    if (locations.length > 0) {
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
    if (this.drawingManager) {
      this.drawingManager.clear();
    }
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
    this.map.setView({ bounds, padding: 0 });
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
    let x1;
    let y1;
    let x2;
    let y2;

    if (shapes && shapes.length > 0) {
      console.log('Retrieved ' + shapes.length + ' from the drawing manager.');
      for (let s of shapes) {
        if (s.data.geometry.type == "Point") {
          coordinates = s.data.geometry.coordinates;

          const lon = coordinates[0];
          const lat = coordinates[1];

          const offset = 0.0000175; // approx ~111m (adjust based on zoom level and context)

          // Define square polygon around the point
          // Compute bounding box corners
          const west = lon - offset;
          const east = lon + offset;
          const north = lat + offset;
          const south = lat - offset;
          x1 = west;
          y1 = north;
          x2 = east;
          y2 = south;
        }
        else {
          x1 = s.geometry.boundingBox.getWest();
          y1 = s.geometry.boundingBox.getNorth();
          x2 = s.geometry.boundingBox.getEast();
          y2 = s.geometry.boundingBox.getSouth();
        }
        // If a circle is drawn, draw a square using the center and radius of the circle
        if (s.properties?.shape === 'Circle' || s.circlePolygon) {

          const centerLng = (x1 + x2) / 2;
          const centerLat = (y1 + y2) / 2;
          const center = [centerLng, centerLat];

          // 3. Estimate radius in meters using getDistanceTo (from center to one edge)
          const pointOnEdge = [x2, centerLat]; // East edge
          const radius = atlas.math.getDistanceTo(center, pointOnEdge); // in meters

          // 4. Use getDestination to find the square corners
          const north = atlas.math.getDestination(center, radius, 0);    // bearing 0° (North)
          const east = atlas.math.getDestination(center, radius, 90);    // 90° (East)
          const south = atlas.math.getDestination(center, radius, 180);  // 180° (South)
          const west = atlas.math.getDestination(center, radius, 270);   // 270° (West)

          // 5. Construct square using those points
          const squareCoords = [[
            [west[0], north[1]],  // Top-left
            [east[0], north[1]],  // Top-right
            [east[0], south[1]],  // Bottom-right
            [west[0], south[1]],  // Bottom-left
            [west[0], north[1]]   // Closing loop
          ]];

          // Step 4: Create a polygon and add to map
          const circleSquarePolygon = new atlas.data.Feature(new atlas.data.Polygon(squareCoords));

          let CircleSquareDataSource = new atlas.source.DataSource();
          CircleSquareDataSource.add(circleSquarePolygon);
          this.map.sources.add(CircleSquareDataSource);

        }
        console.log("Adding " + s.geometry.bounds.toString());


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
            'ct', 1.0, tileId, -1 /*id_in_tile*/, true, true);
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
    if (this.drawingManager) {
      this.drawingManager.clear();
    }
    if (this.entities) {
      this.entities.clear();
    }
  }


  clearAll() {
    if (this.hasShapes) {
      this.clearShapes();
    }

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
    let locs = [];
    if (currentUI.value === 'azure') {
      var circleShape = new atlas.Shape(new atlas.data.Point(center), 'circleShape', {
        subType: "Circle",
        radius
      });
      const existingCircleDataSource = currentMap.map.sources.getById('circleDataSource');
      var dataSource = existingCircleDataSource ?? new atlas.source.DataSource('circleDataSource');
      if (!existingCircleDataSource) {
        currentMap.map.sources.add(dataSource);
      }
      dataSource.add(circleShape);
      currentMap.map.layers.add(new atlas.layer.LineLayer(dataSource, 'circleShapeLayer', {
        strokeColor: 'blue',
        strokeWidth: 3
      }));
      this.points = circleShape.circlePolygon.geometry.coordinates;
      currentMap.boundaries.push(new PolygonBoundary(this.points[0]));
      return;
    }

    if (currentUI.value === 'bing') {

      locs = Microsoft.Maps.SpatialMath.getRegularPolygon(
        new Microsoft.Maps.Location(center[1], center[0]),
        radius,
        256,
        Microsoft.Maps.SpatialMath.DistanceUnits.Meters);
    }

    this.points = locs.map(l => [l.longitude, l.latitude]);
  }
}



//
// PlaceRects - rectangles on the map (results, tiles, bounding boxes)
//

class PlaceRect {

  constructor(x1, y1, x2, y2, color, fillColor, opacity, classname, listener, classtype) {
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
    this.color = color;
    this.fillColor = fillColor;
    this.opacity = opacity;
    this.classname = classname;
    this.classtype = classtype;
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
    if (currentUI.value === 'bing') {
      bingMap.setCenter([(this.x1 + this.x2) / 2, (this.y1 + this.y2) / 2]);
      bingMap.setZoom(19);
    }
    else if (currentUI.value === 'azure') {
      azureMap.setCenter([(this.x1 + this.x2) / 2, (this.y1 + this.y2) / 2]);
      azureMap.setZoom(18);
    }

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

  constructor(x1, y1, x2, y2, metadata, url, tileID, uuid, image_hash, classname, classtype, lat, lng, w, h) {
    super(x1, y1, x2, y2, "#0000FF", "#0000FF", 0.0, "tile", classtype)
    this.metadata = metadata; // for Bing maps
    this.url = url;
    this.tileID = tileID;
    this.uuid = uuid;
    this.image_hash = image_hash;
    this.classname = classname;
    this.classtype = classtype;
    this.lat = lat;
    this.lng = lng;
    this.w = w;
    this.h = h;

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

  constructor(x1, y1, x2, y2, classname, conf, tile, idInTile, inside, selected, secondary, classtype, uuid, image_hash, silverx1, silvery1, silverx2, silvery2) {
    super(x1, y1, x2, y2, conf === 1.0 ? "blue" : "#FF0000", conf === 1.0 ? "blue" : "#FF0000", 0.2, classname, classtype, () => {
      this.highlight(false, true);
    })
    this.conf = conf;
    this.inside = inside;
    this.idInTile = idInTile;
    this.selected = selected;
    this.classname = classname;
    this.classtype = classtype;
    this.address = "";
    this.maxConf = conf; // minimum confidence across same address towers, only recorded in first
    this.firstDet = null; // first of block of same address towers
    this.tile = tile; // id of detection tile
    this.secondary = secondary;

    this.id = Detection_detections.length;
    this.originalId = this.id;
    this.uuid = uuid;
    this.image_hash = image_hash;
    this.silverx1 = silverx1;
    this.silverx2 = silverx2;
    this.silvery1 = silvery1;
    this.silvery2 = silvery2;
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
    let sameAddressInside = "";
    let firstDet = null;
    let boxes = "<ul>";
    let count = 0;
    for (let det of Detection_detections) {
      if ((det.address !== currentAddr) || (det.address === currentAddr && (det.inside != sameAddressInside))) {
        if (currentAddr !== "") {
          boxes += "</ul></li>";
        }
        boxes += "<li id='addrli" + det.id + "'>";
        boxes += "<span class='caret' onclick='";
        boxes += "this.parentElement.querySelector(\".nested\").classList.toggle(\"active\"),";
        boxes += "this.classList.toggle(\"caret-down\")';"
        boxes += "'></span>";
        boxes += "<label id='lbladdrcb" + det.id + "' style='display:none' for='addrcb" + det.id + "'>addrcb" + det.id + "</label>"
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
      if (det.address === currentAddr) {
        sameAddressInside = det.inside;
      }
      else {
        sameAddressInside = "";
      }
      boxes += det.generateCheckBox();
      firstDet.maxConf = Math.max(det.conf, firstDet.maxConf); // record min conf in block header
      firstDet.maxP2 = Math.max(det.p2, firstDet.maxP2)
      det.firstDet = firstDet; // record block header
      det.indexInList = count;
      // det.update();
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
    if (this.selected) {
      this.classtype = 0;
      this.classname = 'ct'
    }
    else {
      this.classtype = 1;
      this.classname = 'not-ct';
    }
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
        if (det.selected) {
          det.classtype = 0;
          det.classname = 'ct'
        }
        else {
          det.classtype = 1;
          det.classname = 'not-ct';
        }
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
    // Do not change the display to 'none' if the main addrli item is displaying as one of the firstdet is visible

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
    super.highlight("#00ff00"); //green
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
      currentMap.clearAll();
      return;
    }
  }

  let engine = $('input[name=model]:checked', '#engines').val()
  let provider = $('input[name=provider]:checked', '#providers').val()
  provider = provider.substring(0, provider.length - 9);


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
    console.log("Estimate request in progress ....");
  } else {
    console.log("Detection request in progress ....");
  }

  // erase the previous set of towers and tiles
  Detection.resetAll();
  // Tile.resetAll();

  // first, play the request, but get an estimate of the number of tiles
  const formData = new FormData();
  formData.append('bounds', bounds);
  formData.append('engine', engine);
  formData.append('provider', provider);
  formData.append('polygons', boundaries);
  formData.append('estimate', "yes");

  fetch("/getobjects", { method: "POST", body: formData, })
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
      startTime = performance.now();

      // now, the actual request
      console.log("Detecting Cooling Towers ....");
      Detection.resetAll();
      formData.delete("estimate");
      fetch("/getobjects", { method: "POST", body: formData })
        .then(response => response.json())
        .then(result => {
          // // Need to add code to process the results from EDAV
          // disableProgress(0, 0);
          // return;
          console.log("Processing ....");
          processObjects(result, startTime);
        })
        .catch(e => {
          console.log(e + ": "); disableProgress(0, 0);
        });
    });

  getazmapTransactioncountjs(2);
  disableProgress((performance.now() - startTime) / 1000, Tile_tiles.length);
}

function ProcessUserRequest(estimate) {
  try {
    const now = new Date(); // Get current date and time
    document.getElementById('lblSearchEndTime').style.display = "none";
    document.getElementById('lblSearchStartTime').innerHTML = "Last Search Start Time: <br>" + now.toLocaleString();
    document.getElementById('lblSearchStartTime').removeAttribute("style");
    if (Detection_detections.length > 0) {
      if (!window.confirm("This will erase current detections. Proceed?")) {
        // erase the previous set of towers and tiles
        currentMap.clearAll();
        return;
      }
    }

    let engine = $('input[name=model]:checked', '#engines').val()
    let provider = $('input[name=provider]:checked', '#providers').val()
    provider = provider.substring(0, provider.length - 9);

    if (currentMap.hasShapes()) {
      // If user drew the boundaries
      drawnBoundary();
      currentMap.clearCustomBoundaryShapes();
    }
    // now get the boundaries ready to ship
    let bounds = currentMap.getBoundsUrl();

    if (currentMap.boundaries.length === 0) {
      if (currentMap.hasShapes()) {
        drawnBoundary();
      }
    }


    let boundaries = currentMap.getBoundariesStr();
    if (estimate) {
      console.log("Estimate request in progress ....");
    } else {
      console.log("Detection request in progress ....");
    }

    // erase the previous set of towers and tiles
    Detection.resetAll();
    // Tile.resetAll();

    // first, play the request, but get an estimate of the number of tiles
    formData = new FormData();
    formData.append('bounds', bounds);
    formData.append('engine', engine);
    formData.append('provider', provider);
    formData.append('polygons', boundaries);
    formData.append('estimate', "yes");


    fetch("/uploadTileImages", {
      method: "POST", body: formData,
      credentials: 'include' // 👈 keeps cookies/session consistent 
    })
      .then(result => result.text())
      .then(result => {
        if (Number(result) === -1) {
          fatalError("Tile limit for this session exceeded. Please close browser to continue.")
          return;
        }
        console.log("Number of tiles: " + result + ", estimated time: "
          + (Math.round(Number(result) * secsPerTile * 10) / 10) + " s");
        // Get from Dev  Key Vault by default. Need to create secrets in the Prod Key Vault


        // console.log("  Estimated tiles:" + nt);
        if (estimate) {
          return;
        }

        // actual retrieval process starts here
        nt = Number(result);
        enableProgress(nt);
        setProgress(0);
        startTime = performance.now();

        // now, the actual request

        Detection.resetAll();
        formData.delete("estimate");
        fetch("/uploadTileImages", {
          method: "POST", body: formData,
          credentials: 'include' // 👈 keeps cookies/session consistent 
        })
          .then(response => response.json())
          .then(result => {
            console.log("Images uploaded ....");
            formData.append('user_id', result.user_id);
            formData.append('request_id', result.request_id);
            formData.append('tiles_count', result.tiles_count);
            console.log("Polling Cluster Status");

            pollClusterStatusjs();
            // console.log("Delaying Polling Silver Table by 1 minute");
            // setTimeout(pollSilverTableWithLogs, 60000);  // Start polling after 1 minute
          })
          .catch(e => {
            console.log(e + ": "); disableProgress(0, 0);
          });
      });

    getazmapTransactioncountjs(2);
  } catch (error) {
    console.error('Error during main ProcessRequest:', error);
    disableProgress(0, 0);
  }

}


async function fetchWithTimeout(url, options, timeoutDuration) {
  const controller = new AbortController();  // Create an AbortController instance
  const timeoutId = setTimeout(() => controller.abort(), timeoutDuration);  // Set the timeout for the request

  try {
    // Send the HTTP request with the AbortController signal
    const response = await fetch(url, { ...options, signal: controller.signal });

    // If the request was successful, parse the response
    const data = await response.json();
    console.log('Data received:', data);

    // Clear the timeout when the request completes
    clearTimeout(timeoutId);

    return data; // Return the response data
  } catch (error) {
    if (error.name === 'AbortError') {
      console.log('fetchWithTimeout - Request timed out');
      throw new Error("fetchWithTimeout - Request timed out");
    } else {
      throw new Error(`fetchWithTimeout - Fetch error: ${error.text}`);
    }
  }
}

async function pollSilverTable() {
  console.log("Started Polling Silver Table for detections....");
  const url = '/pollSilverTable';  // Endpoint URL
  const options = { method: 'POST', body: formData };  // Request body

  const TIMEOUT_DURATION = 4.9 * 60 * 1000;  // 4.9 minutes in milliseconds
  const RESTART_DELAY = 60000;  // Restart delay in milliseconds (e.g., 60 seconds)
  try {
    while (true) {
      // console.log('Making request...');
      const result = await fetchWithTimeout(url, options, TIMEOUT_DURATION);
      if (result.status === 502) {
        console.log('Error during pollSilverTable request:' + error);
        // In case of error, wait for 10 seconds before retrying
        console.log('Waiting for 60 seconds before retrying...');
        await new Promise(resolve => setTimeout(resolve, RESTART_DELAY));
        return pollSilverTable();
      }
      if (result) {
        console.log('Polling complete. Detections found in the Silver Table. Request completed successfully');
        console.log("Reverse geocoding and drawing bounding boxes ....");
        drawBoundingBoxes();
        return true;
      } else {
        console.log('Polling Silver Table - Request failed or timed out');
      }

      // Wait before sending another request (restart cycle after 10 seconds)
      console.log('Waiting for 10 seconds before retrying...');
      await new Promise(resolve => setTimeout(resolve, RESTART_DELAY));
      return pollSilverTable();

    }
  } catch (error) {
    console.log('Error during pollSilverTable request:' + error);
    // In case of error, wait for 10 seconds before retrying
    console.log('Waiting for 10 seconds before retrying...');
    await new Promise(resolve => setTimeout(resolve, RESTART_DELAY));
    return pollSilverTable();
  }
}
async function pollSilverTableWithLogs() {
  console.log("Started Polling Silver Table for detections....");
  const url = '/pollSilverTableWithLogs';  // Endpoint URL
  const TIMEOUT_DURATION = 4.9 * 60 * 1000;  // 4.9 minutes in milliseconds
  const controller = new AbortController();  // Create an AbortController instance
  let timeoutHandle;
  const params = new URLSearchParams(formData).toString();
  const tilecount = formData.get('tiles_count');
  var RESTART_DELAY = 60000;  // Restart delay in milliseconds (e.g., 60 seconds)
  if (tilecount < 100) {
    RESTART_DELAY = tilecount * 5000;  // Restart delay in milliseconds (e.g., 5 seconds per tile)
  }

  try {



    const eventSource = new EventSource(`/pollSilverTableWithLogs?${params}`);
    let completed = false;

    eventSource.onmessage = (event) => {
      console.log('📩 Update: ' + event.data);
    };
    eventSource.addEventListener('done', (event) => {
      completed = true;

      if (timeoutHandle) clearTimeout(timeoutHandle); // Clear any pending timeouts  // 🧼 stop the auto-reconnect
      eventSource.close();
      console.log('Polling complete.' + event.data);
      console.log("Reverse geocoding and drawing bounding boxes ....");
      drawBoundingBoxes();

      return true;


    });
    timeoutHandle = setTimeout(async () => {
      if (!completed && eventSource.readyState !== EventSource.CLOSED) {
        console.log('⏱️ Closing connection after 5 minutes...');
        eventSource.close();


        console.log('⏱️ Reconnecting after 4.9 minutes...');
        // Wait before sending another request (restart cycle after 10 seconds)
        console.log('Waiting for 60 seconds before retrying...');
        await new Promise(resolve => setTimeout(resolve, RESTART_DELAY));
        return pollSilverTableWithLogs();

      } // 4.9 minutes
    }, TIMEOUT_DURATION);

    eventSource.onerror = (error) => {
      console.error('SSE connection error Polling Silver Table:', error);
      eventSource.close();
      // Wait before sending another request (restart cycle after 10 seconds)
      console.log('Waiting for 10 seconds before retrying...');
      // Wait 10 seconds before starting the next process
      setTimeout(() => {
        console.log('Waiting for 10 seconds before retrying...');
        return pollSilverTableWithLogs();
      }, 10000); // 10,000 ms = 10 seconds
    };




  } catch (error) {
    console.log('Error during pollSilverTableWithLogs request:' + error);
    // In case of error, wait for 10 seconds before retrying
    eventSource.close();
    console.log('Waiting for 60 seconds before retrying...');
    await new Promise(resolve => setTimeout(resolve, RESTART_DELAY));
    return pollSilverTableWithLogs();
  }
}


async function pollquerystatus(startTime) {

  console.log("Checking the merge query execution status....");
  const url = '/pollquerystatus';  // Endpoint URL
  const options = { method: 'POST', body: formData };  // Request body
  const TIMEOUT_DURATION = 4.9 * 60 * 1000;  // 4.9 minutes in milliseconds
  const RESTART_DELAY = 60000;  // Restart delay in milliseconds (e.g., 60 seconds)
  try {
    while (true) {
      // console.log('Making request...');
      const result = await fetchWithTimeout(url, options, TIMEOUT_DURATION);
      if (result.status === 502) {
        console.log('Error during Polling merge query status:' + error);
        // In case of error, wait for 60 seconds before retrying
        console.log('Waiting for 60 seconds before retrying...');
        await new Promise(resolve => setTimeout(resolve, RESTART_DELAY));
        return pollquerystatus();
      }
      if (result) {
        console.log('Merge completed successfully for ' + Detection_detections.length.toString() + ' detections.');
        // disableProgress(0,0);
        return true;
      } else {
        console.log('Polling Polling merge query status - Request failed or timed out');
      }

      // Wait before sending another request (restart cycle after 60 seconds)
      console.log('Waiting for 60 seconds before retrying...');
      await new Promise(resolve => setTimeout(resolve, RESTART_DELAY));
      return pollquerystatus();

    }
  } catch (error) {
    console.log('Error during pollquerystatus request:' + error);
    // In case of error, wait for 10 seconds before retrying
    console.log('Waiting for 60 seconds before retrying...');
    await new Promise(resolve => setTimeout(resolve, RESTART_DELAY));
    return pollquerystatus();
  }
}


// Function to restart the request cycle after each attempt
function restartRequest() {
  setTimeout(pollSilverTable, 10000);  // Restart the cycle every 10 seconds
}
// Function to get the bounding box (East, West, North, South) of the polygon
function getBoundingBox(polygon) {
  let west = Infinity;  // Initialize to very large number
  let east = -Infinity; // Initialize to very small number
  let north = -Infinity; // Initialize to very small number
  let south = Infinity; // Initialize to very large number

  // Loop through the coordinates of the polygon
  polygon.coordinates.forEach(coord => {
    const longitude = coord[0]; // Longitude is at index 0
    const latitude = coord[1];  // Latitude is at index 1

    // Update the bounding box
    if (longitude < west) {
      west = longitude; // Westmost longitude
    }
    if (longitude > east) {
      east = longitude; // Eastmost longitude
    }
    if (latitude > north) {
      north = latitude; // Northmost latitude
    }
    if (latitude < south) {
      south = latitude; // Southmost latitude
    }
  });

  return { west, east, north, south };
}
function drawBoundingBoxes() {
  try {
    formData.delete("estimate");
    fetch("/fetchBoundingBoxResults", { method: "POST", body: formData })
      .then(response => response.json())
      .then(result => {
        console.log("Processing ....");
        processObjects(result, startTime);
      })
      .catch(e => {
        console.log(e + " - drawBoundingBoxes: "); disableProgress(0, 0);
      });
  }
  catch (error) {
    console.log('Error during drawBoundingBoxes:', error);

  }
}
// // Start the cycle when the script first loads
// checkCondition();

//get Azure map transaction count for the current month
function getazmapTransactioncountjs(intEnv) {
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

//get Cluster Status
async function getClusterStatusjs() {

  const response = await fetch('/getClusterStatus');
  const message = await response.text();
  const now = new Date(); // Get current date and time
  document.getElementById('lblClusterStatusValue').innerText = message;
  document.getElementById('lblClusterStatusDtTime').innerText = now.toLocaleString();
  document.getElementById('lblClusterStatusDtTime').removeAttribute("style");
  document.getElementById('btnGetClusterStatus').removeAttribute("style");
  try {
    if (message === 'RUNNING') {
      document.getElementById('lblClusterStatusValue').style.backgroundColor = "green";
      document.getElementById('lblClusterStatusValue').style.color = "white";
    } else if (message === 'PENDING') {
      document.getElementById('lblClusterStatusValue').style.backgroundColor = "blue";
      document.getElementById('lblClusterStatusValue').style.color = "white";
    } else if (message === 'RESIZING') {
      document.getElementById('lblClusterStatusValue').style.backgroundColor = "yellow";
      document.getElementById('lblClusterStatusValue').style.color = "black";
    } else {
      document.getElementById('lblClusterStatusValue').style.backgroundColor = "red";
      document.getElementById('lblClusterStatusValue').style.color = "white";
    }

    return message;
  } catch (error) {
    console.error('Error fetching cluster status:', error);
    return null; // or throw error, depending on your needs
  }

}
function ToggleTestEnvironment() {
  let enableTestEnv = fetch('/ToggleTestEnvironment');
  fetch('/ToggleTestEnvironment')
    .then(response => {

      return response.text();
    })
    .then(data => {
      // Assign the response value to a variable
      enableTestEnv = data;
      if (enableTestEnv === 'True') {
        document.getElementById("btnTest").removeAttribute("style");
        document.getElementById("btnTestGold").removeAttribute("style");
      }

    })
    .catch(error => {
      console.log(error);
    });


}
//get Cluster Status
async function pollClusterStatusjs() {
  let isFirstAttempt = true;
  const TIMEOUT_DURATION = 4.9 * 60 * 1000;  // 4.9 minutes in milliseconds
  let timeoutHandle;  // Need to clear this before returning true
  try {
    // while (true) {

    const result = await getClusterStatusjs();

    if (result == 'RUNNING') {
      console.log('Cluster is Running');
      if (timeoutHandle) clearTimeout(timeoutHandle); // Clear any pending timeouts
      if (isFirstAttempt) {
        isFirstAttempt = false; // Mark that we've handled the first attempt
        const tilecount = formData.get('tiles_count');
        var RESTART_DELAY = 60000;  // Restart delay in milliseconds (e.g., 60 seconds)
        if (tilecount < 100) {
          RESTART_DELAY = tilecount * 5000;  // Restart delay in milliseconds (e.g., 5 seconds per tile)
        }

        console.log("Delaying polling of Silver Table by " + Math.round((RESTART_DELAY / 1000), 2) + " seconds.");

        setTimeout(pollSilverTableWithLogs, RESTART_DELAY); // Wait 1 minute, then call
      } else {
        return pollSilverTableWithLogs(); // Immediately call on non-first attempts
      }
    } else {

      console.log('Cluster status: ' + result);
      console.log('Waiting for 5 minutes before retrying...');
      isFirstAttempt = false; // First attempt failed; no need to delay on next RUNNING
      timeoutHandle = setTimeout(() => {
        pollClusterStatusjs(); // Recursive call after timeout
      }, TIMEOUT_DURATION);
    }




  } catch (error) {

    // In case of error, wait for 10 seconds before retrying
    console.log('Error checking Cluster Status...');
    console.log('Waiting for 5 minutes before retrying...');
    isFirstAttempt = false; // First attempt failed; no need to delay on next RUNNING
    timeoutHandle = setTimeout(() => {
      pollClusterStatusjs(); // Recursive call after timeout
    }, TIMEOUT_DURATION);
  }

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
    if (r['class'] === 0) { // detection
      let det = new Detection(r['x1'], r['y1'], r['x2'], r['y2'],
        r['class_name'], r['conf'], r['tile'], r['id_in_tile'], r['inside'], r['selected'], r['secondary'], r['class'], r['uuid'], r['image_hash'],
        r['silverx1'], r['silvery1'], r['silverx2'], r['silvery2']);
    } else if (r['class'] === 10) { // tile
      let tile = new Tile(r['x1'], r['y1'], r['x2'], r['y2'], r['metadata'], r['url'], r['id'], r['uuid'], r['image_hash'], r['class_name'], r['class'], r['lat'], r['lng'], r['w'], r['h']);
    }
  }
  //console.log("" + Detection_detections.length + " detections.")

  augmentDetections();

  disableProgress((performance.now() - startTime) / 1000, Tile_tiles.length);
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

    currentMap.showBoundaries();
  }
}

function drawnBoundary() {
  console.log("using custom boundary polygon(s)");


  // // Clear existing boundaries
  // currentMap.clearAll();
  // Draw boundary
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
      currentProvider = 'azure';//rad; //Bing Maps is the defacto provider for now

      // and one for the file input box
      let fileBox = document.getElementById("upload_file");
      fileBox.addEventListener('change', () => {
        uploadImage();
      });

      // also add change listeners for the UI providers
      currentUI = document.uis.uis[0];
      // setMap(currentUI);
      for (let rad of document.uis.uis) {
        rad.addEventListener('change', function () {
          setMap;
        });
      }

    }
  });
}

function setMap(newMap = currentMap) {

  if (currentUI !== null) {
    currentMap.clearAll();
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
    document.getElementById("azureSearchBoxContainer").style.display = "none";
    document.getElementById("bingSearchBoxContainer").style.display = "inline";
    initBingMap();
    zoom = currentMap.getZoom();
    center = currentMap.getCenter();
  } else if (currentUI.value === "azure") {
    document.getElementById("uploadsearchui").style.display = "none";
    document.getElementById("mapsearchui").style.display = null;
    document.getElementById("fdetect").style.display = null;
    document.getElementById("ftowers").style.display = null;
    document.getElementById("fsave").style.display = null;
    document.getElementById("freview").style.display = null;
    document.getElementById("fadd").style.display = null;
    document.getElementById("bingSearchBoxContainer").style.display = "none";
    document.getElementById("azureSearchBoxContainer").style.display = "inline";
    currentMap = azureMap;

    // recreate boundaries for azure
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
function chunkArray(array, chunkSize) {
  const result = [];
  for (let i = 0; i < array.length; i += chunkSize) {
    result.push(array.slice(i, i + chunkSize));
  }
  return result;
}
function augmentDetections(addnew = false) {
  if (addnew != true) {
    Detection_detectionsAugmented = 0;
  }

  //for (let det of Detection_detections) {
  if (currentUI.value == "bing") {

    for (let i = 0; i < Detection_detections.length; i++) {
      let det = Detection_detections[i];
      if (det.address !== "") {
        Detection_detectionsAugmented++;
        continue;
      }
      let loc = det.getCenterUrl();
      // call Bing maps api instead at:
      setTimeout((ix) => {
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
      }, 1000 * i, i)
    }

  }
  else if (currentUI.value == "azure") {
    const BATCH_SIZE = 99;
    let batchItems = [];
    if (Detection_detections.length == 0) {
      console.log("Done. No detections found.");
      const now = new Date(); // Get current date and time
      document.getElementById('lblSearchEndTime').innerHTML = "Last Search End Time: <br>" + now.toLocaleString();
      document.getElementById('lblSearchEndTime').removeAttribute("style");
    }
    else {
      if (CurrentSearcharea === 'nonrural') {//(addresstype === "urban"){
        for (let i = 0; i < Detection_detections.length; i++) {
          let det = Detection_detections[i];
          if (det.address === "") {
            let loc = det.getCenterUrl();
            let reverseloc = loc.split(",")[1] + "," + loc.split(",")[0];
            let coordinates = reverseloc.split(",").map(Number);
            batchItems.push({
              coordinates: coordinates,
              resultTypes: ["Address", "Neighborhood", "PopulatedPlace", "Postcode1", "AdminDivision1", "AdminDivision2", "CountryRegion"],
              OptionalID: det.id
            });
          }
        }


        const batchedItems = chunkArray(batchItems, BATCH_SIZE);
        const existingdetectionslength = Detection_detections.length;
        batchedItems.forEach((batch, batchIndex) => {
          setTimeout(() => {
            $.ajax({
              url: "https://atlas.microsoft.com/reverseGeocode:batch?api-version=2023-06-01&subscription-key=" + azure_api_key,
              type: 'POST',
              contentType: 'application/json',
              data: JSON.stringify({ batchItems: batch }),
              success: async function (result) {
                for (let i = 0; i < result.batchItems.length; i++) {
                  const features = result.batchItems[i].features;
                  if (features && features.length > 0) {
                    const addr = features[0].properties.address.formattedAddress;
                    let detectionIndex = batchIndex * 99 + i;
                    if (addnew == true) {
                      // detectionIndex = Detection_detections.findIndex(item => item.id === batchItems[i].OptionalID);
                      detectionIndex = result.batchItems[i]["optionalId"];
                    }
                    Detection_detections[detectionIndex].augment(addr);


                  } else {
                    if (result.batchItems[i]['error'] != "") {
                      console.log(`${result.batchItems[i]['error'].code} at ${batchIndex * 99 + i}. Please try again`);
                      return;
                    }

                    console.log(`No address found for batch item ${batchIndex * 99 + i}`);
                  }
                }

                afterAugment();
              },
              error: function (error, status) {
                if (error.status == 500) {
                  console.error("Internal Error in batch reverse geocode. Please try again.", error);
                }

              }
            });
          }, 1000 * batchIndex);
        });
      }
      else { //search area is rural
        for (let i = 0; i < Detection_detections.length; i++) {
          let det = Detection_detections[i];
          if (det.address === "") {
            let loc = det.getCenterUrl();
            let [lon, lat] = loc.split(",").map(Number).reverse(); // swap to lat,lon
            // Call simple reverse API

            try {
              // call Bing maps api instead at:
              const url = `https://atlas.microsoft.com/search/address/reverse/json` +
                `?api-version=1.0` +
                `&query=${lat},${lon}` +
                `&subscription-key=${azure_api_key}` +
                `&includeEntityTypes=Address,Neighborhood, PopulatedPlace, Postcode1, AdminDivision1, AdminDivision2, CountryRegion` +
                `&output=json`;
              setTimeout((ix) => {
                //console.log(ix+1);
                fetch(url)
                  .then(res => res.json())
                  .then(data => {
                    const address = data.addresses?.[0]?.address?.freeformAddress || 'No address found';

                    det.augment(address);
                    afterAugment();
                    // console.log(`[${i + 1}] → ${address}`);
                  })
                  .catch(err => {
                    // results.push({ lat, lon, address: 'ERROR' });
                    console.error(`[${i + 1}] Error:`, err);
                  });
              }, 1000 * i);
              

            } catch (err) {
              console.error(`Error retrying simple reverse geocode:`, err);
              ToggleSearchArea('nonrural');
            }

          }
        }

      }
    }

  }
}
function afterAugment() {
  // wait for the last one
  if (Detection_detectionsAugmented !== Detection_detections.length) {
    return;
  }

  Detection.sort();
  console.log("Generating List of detections ..... ")
  Detection.generateList();

  // now hide low confidence values, sort the list and do the rest

  console.log("Adjusting Confidence ....");
  adjustConfidence();
  const now = new Date(); // Get current date and time
  document.getElementById('lblSearchEndTime').innerHTML = "Last Search End Time: <br>" + now.toLocaleString();
  document.getElementById('lblSearchEndTime').removeAttribute("style");
  console.log("Done.");
  ToggleSearchArea('nonrural');
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


function PromoteSilverToGold() {
  try {
    nd = Detection_detections.length
    if (nd > 0) {
      enableProgress(nd);
      setProgress(0);
      startTime = performance.now();
      console.log("Building detections List with updated selected/unselected values...")
      tileRecords = [];
      //Loop through the Tiles
      for (let Tile of Tile_tiles) {

        // Filter detections for the tile using uuid and image_hash

        DetectionsForTile = Detection_detections.filter(item => (item.uuid === Tile.uuid) && (item.image_hash === Tile.image_hash));
        bboxes = [];
        // Looping through the updated(selected/unselected) detection array elements
        for (let det of DetectionsForTile) {
          // Exclude only newly added items
          // if (det.idInTile !== -1) {
          bboxes.push({
            'uuid': det.uuid,
            'image_hash': det.image_hash,
            'conf': det.conf,
            'class': det.classtype,
            'x1': det.silverx1,
            'x2': det.silverx2,
            'y1': det.silvery1,
            'y2': det.silvery2,
            'class_name': det.classname,
            'secondary': det.secondary,
          });
          //console.log(" including detection #" + (det.originalId));
          // }

        }
        tileRecords.push({
          'uuid': Tile.uuid,
          'image_hash': Tile.image_hash,
          'bboxes': bboxes,

        })
      }
      formData.delete("SilverDetections");
      formData.delete("bounds");
      formData.delete("boundaries");
      formData.delete("include");
      formData.delete("additions");
      formData.append("SilverDetections", JSON.stringify(tileRecords));
      console.log('Promoting ' + nd.toString() + ' detections to the Gold Table......');
      // Submit request to excute query and get the statement_id
      fetch('/promoteSilverToGold', { method: "POST", body: formData })
        .then(response => response.json())
        .then(data => {

          if (data['SQLstatement_id'] != "") {
            formData.delete("SQLstatement_id");
            formData.append("SQLstatement_id", data['SQLstatement_id']);
            // wait for 10 seconds before starting the poll 

            setTimeout(pollquerystatus(startTime), 10000);
            disableProgress((performance.now() - startTime) / 1000, nd);
            // console.log('Detections successfully promoted to the Gold Table.'); 
          }
          else {
            disableProgress(0, 0);
            throw new Error(`Silver to Gold promotion was not successull`);

          }
        })
        .catch(error => {
          console.log(error);
          disableProgress(0, 0);
        });

    }
  }
  catch (error) {
    console.log('An error occurred PromoteSilverToGold: ' + error);
    disableProgress(0, 0);
  }
  finally {

  }
}


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
    img.addEventListener('load', () => { removeCustomImage(url); }, { once: true });
  }
}

function removeCustomImage(url) {
  fetch('/rm' + url, { method: "GET" });
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
let secsPerTile = 0.6;
let dataPoints = 0;

function enableProgress(tiles) {
  document.getElementById("progress_div").style.display = "flex";

  progressTimer = setInterval(progressFunction, 100);
  numTiles = tiles;
  if (numTiles < 100) {
    totalSecsEstimated = secsPerTile * numTiles;
  }
  else {
    totalSecsEstimated = 60;
  }

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
  return geom['type'] === 'Polygon' ? [coords] : coords;
}

// init actions
console = new myConsole();

fillProviders();
confSlider.value = Math.round(Detection_minConfidence * 100);

// Function to assign aria-labels to elements that need them
function assignAriaLabels() {
  // Select specific elements that typically need aria-labels
  const elementsNeedingAriaLabel = document.querySelectorAll('button, a, input, textarea, select, checkbox');


  // Loop through each element
  elementsNeedingAriaLabel.forEach(element => {
    const id = element.id; // Get the element's ID
    if (!element.hasAttribute('aria-label')) {

      element.setAttribute('aria-label', element.id);
      // console.log(`Assigned aria-label to ${id}: ${element.id}`);

    }
  });
  const mapelementsNeedingAriaLabel = document.getElementById("bingMap").querySelectorAll('*');
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
function ToggleSearchArea(searcharea){
  if (searcharea==='rural'){
    document.getElementById('rural').checked = true;
    document.getElementById('nonrural').checked = false;
    CurrentSearcharea = 'rural';
  }
  else{
    document.getElementById('nonrural').checked = true;
    document.getElementById('rural').checked = false;
    CurrentSearcharea = 'nonrural';
  }
}
function addEventListenertosearcharea(){
  let selectedRadio = null;
  const radios = document.querySelectorAll('input[type="nonrural"][name="rural"]');

  radios.forEach(radio => {
    radio.addEventListener('click', function () {
      if (selectedRadio === this) {
        this.checked = false;
        selectedRadio = null;
      } else {
        selectedRadio = this;
      }
    });
  });
}
document.addEventListener('DOMContentLoaded', function () {
  const menu = document.getElementById('mapContextMenu');
  menu.addEventListener('click', function () {
    const lat = this.dataset.lat;
    const lng = this.dataset.lng;
    const coordsText = `${lat}, ${lng}`;

    navigator.clipboard.writeText(coordsText).then(() => {
      console.log(`Coordinates copied:${coordsText}`);
    });

    this.style.display = 'none';
  });
});

console.log("TowerScout initialized.");
