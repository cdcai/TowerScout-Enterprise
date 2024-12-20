const mockData = {
    "84124": {
        "additionalData": [
            {
                "providerID": "00005555-5400-2800-0000-00000bf3900f",
                "geometryData": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [
                                            -111.8655481,
                                            40.6748948
                                        ],
                                        [
                                            -111.8655462,
                                            40.6746936
                                        ],
                                        [
                                            -111.8655451,
                                            40.6745705
                                        ],
                                        [
                                            -111.8655481,
                                            40.6748948
                                        ]
                                    ]
                                ]
                            },
                            "id": "00005555-5400-2800-0000-00000bf3900f"
                        }
                    ]
                }
            }
        ]
    },
    "23456": {
        "additionalData": [
            {
                "providerID": "00005556-4100-2800-0000-00000c0548cb",
                "geometryData": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {},
                            "geometry": {
                                "type": "MultiPolygon",
                                "coordinates": [
                                    [
                                        [
                                            [
                                                -76.163787,
                                                36.760132
                                            ],
                                            [
                                                -76.16218,
                                                36.759906
                                            ],
                                        ]
                                    ],
                                    [
                                        [
                                            [
                                                -75.982729,
                                                36.562498
                                            ],
                                            [
                                                -75.982642,
                                                36.560248
                                            ],
                                            [
                                                -75.982371,
                                                36.559969
                                            ],
                                            [
                                                -75.982102,
                                                36.559894
                                            ],
                                        ]
                                    ],
                                    [
                                        [
                                            [
                                                -75.970901,
                                                36.660753
                                            ],
                                            [
                                                -75.970881,
                                                36.660692
                                            ],
                                            [
                                                -75.97081,
                                                36.660657
                                            ],
                                            [
                                                -75.970901,
                                                36.660753
                                            ]
                                        ]
                                    ],
                                    [
                                        [
                                            [
                                                -75.969473,
                                                36.664585
                                            ],
                                            [
                                                -75.969438,
                                                36.664558
                                            ],
                                            [
                                                -75.969331,
                                                36.664484
                                            ],
                                            [
                                                -75.969473,
                                                36.664585
                                            ]
                                        ]
                                    ],
                                ]
                            },
                            "id": "00005556-4100-2800-0000-00000c0548cb"
                        }
                    ]
                }
            }
        ]
    }
};

  function testFetchGeometry(geometryId) {
    const geometryData = mockData[geometryId];
    if (geometryData) {
        const geometryType = geometryData.additionalData[0].geometryData.features[0].geometry.type;
        if (geometryType === "MultiPolygon") {
            const result = geometryId==="23456";
            console.log(`${result ? "PASSED: " : "FAILED: " } GeometryId=23456: ${result} and geometryType=${geometryType}`)
            return;
        }

        if (geometryType === "Polygon") {
            const result = geometryId==="84124";
            console.log(`${result ? "PASSED: " : "FAILED: " } GeometryId=84124: ${result} and geometryType=${geometryType}`)
            return;
        }
    }
    console.log(`FAILED: Geometry data not available.`)
  }

  // Test the function with both geometry IDs
  testFetchGeometry("84124"); // Test with Polygon
  testFetchGeometry("23456");  // Test with MultiPolygon
