from databricks import sql
from requests.exceptions import Timeout, RequestException
import os, time, logging
from databricks.sql.exc import RequestError


class SilverTable:
    def __init__(self):
        self.batch_size = 100
        
    # @retry(wait=wait_exponential(min=2, max=10), stop=stop_after_attempt(3))
    def fetchbboxes(self, request_id, user_id, tile_count):
        logging.info("Startinng ts_readdetections.py(fetchbboxes)")
        attempt = 0
        retries = 3
        connection = None
        cursor = None
        while attempt < retries:
            try:
                max_retries = tile_count * 2
                jobdone = self.poll_SilverTableJobDone(
                    request_id, user_id, tile_count, max_retries
                )
                # jobdone = poll_table_for_record_count(
                # request_id, user_id, tile_count, max_retries
                # )
                if jobdone:
                    connection = sql.connect(
                        server_hostname="adb-1881246389460182.2.azuredatabricks.net",
                        http_path="/sql/1.0/warehouses/8605a48953a7f210",
                        access_token="dapicb010df06931117a00ccc96cab0abdf0-3",
                        connection_timeout= 600,  # Timeout in seconds
                        socket_timeout=600,
                        retry_config={
                         "min_retry_delay": 1.0,
                         "max_retry_delay": 60.0,
                         "max_attempts": max_retries,
                         "retry_duration": 900.0,
                         "default_retry_delay": 5.0,
                        }
                       
                    )
                # # Testing existing data
                # user_id = 'cnu4'
                # request_id = '32c219ef' # azure 10 mtrs
                # user_id = 'f39e5f58-9d40-4c06-8665-dd5415fa1ed3'#azure app services
                # request_id = 'e6395b66' #azure app services 500mtrs
            
                    cursor = connection.cursor()

                    cursor.execute(
                        "SELECT bboxes from edav_dev_csels.towerscout.test_image_silver WHERE user_id = '"
                        + user_id
                        + "' AND request_id = '"
                        + request_id
                        + "' order by image_id"
                    )
                    result = cursor.fetchall()

                # print(result)

                    cursor.close()
                    connection.close()
                    return result
            except sql.InterfaceError as e:
                logging.error(
                    "Interface Error at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                )
                cursor.close()
                connection.close()
            except RequestError as e:
                logging.error(
                    "RequestError at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                    )
                attempt += 1
                if attempt < retries:
                    logging.info(f"Retrying {attempt}/{retries} in 10 seconds...")
                    time.sleep(10) 
                    retries+=1
            except RequestException as e:
                logging.error(
                    "RequestException at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                    )
                attempt += 1
                if attempt < retries:
                    logging.info(f"Retrying {attempt}/{retries} in 10 seconds...")
                    time.sleep(10) 
                    retries+=1 
            except sql.DatabaseError as e:
                logging.error(
                    "Database Error at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                )
                attempt += 1
                if attempt < retries:
                    logging.info(f"Retrying {attempt}/{retries} in 10 seconds...")
                    time.sleep(10) 
                    retries+=1 
            except (Timeout, ConnectionError) as e:
                logging.error(
                    "Timeout,ConnectionError at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                )
                attempt += 1
                if attempt < retries:
                    logging.info(f"Retrying {attempt}/{retries} in 10 seconds...")
                    time.sleep(10) 
                    retries+=1 
            except sql.OperationalError as e:
                logging.error(
                    "Operational Error at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                )
                cursor.close()
                connection.close()
            except Exception as e:
                cursor.close()
                connection.close()
                logging.error(
                    "Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e
                )
            except RuntimeError as e:
                cursor.close()
                connection.close()
                logging.error("Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e)
            except SyntaxError as e:
                cursor.close()
                connection.close()
                logging.error("Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e)
            finally:
                if cursor:
                    cursor.close()
                if connection:
                    connection.close()
    
    def fetchbboxesWithoutPolling(self, request_id, user_id, tiles_count):
        logging.info("Startinng ts_readdetections.py(fetchbboxes)")
        attempt = 0
        retries = 3
        connection = None
        cursor = None
        while attempt < retries:
            try:
                max_retries = tiles_count * 2
                jobdone = True
                # jobdone = poll_table_for_record_count(
                # request_id, user_id, tile_count, max_retries
                # )
                if jobdone:
                    connection = sql.connect(
                        server_hostname="adb-1881246389460182.2.azuredatabricks.net",
                        http_path="/sql/1.0/warehouses/8605a48953a7f210",
                        access_token="dapicb010df06931117a00ccc96cab0abdf0-3",
                        connection_timeout= 600,  # Timeout in seconds
                        socket_timeout=600,
                        retry_config={
                         "min_retry_delay": 1.0,
                         "max_retry_delay": 60.0,
                         "max_attempts": max_retries,
                         "retry_duration": 900.0,
                         "default_retry_delay": 5.0,
                        }
                       
                    )
                # # Testing existing data
                # user_id = 'cnu4'
                # request_id = '32c219ef' # azure 10 mtrs
                # user_id = 'f39e5f58-9d40-4c06-8665-dd5415fa1ed3'#azure app services
                # request_id = 'e6395b66' #azure app services 500mtrs
                # request_id = '00a6f5bc'
                    cursor = connection.cursor()

                    cursor.execute(
                        "SELECT bboxes, concat(uuid,'.jpeg') AS filename, image_path AS url from edav_dev_csels.towerscout.test_image_silver WHERE user_id = '"
                        + user_id
                        + "' AND request_id = '"
                        + request_id
                        + "' order by image_id"
                    )
                    # query =  """
                    #     SELECT bboxes, named_struct('lat', image_metadata.lat,
                    #     'lat_for_url',image_metadata.lat,
                    #     'lng', image_metadata.long,
                    #     'h', image_metadata.height,
                    #     'w', image_metadata.width,
                    #     'id', image_id,
                    #     'col',0,
                    #     'row',0,
                    #     'url',image_path,
                    #     'filename', concat(uuid, '.jpeg')

                    #     ) as tiles
                    #     FROM edav_dev_csels.towerscout.test_image_silver where request_id = ? AND user_id = ?
                    #     """
                    # cursor.execute(query, (request_id, user_id))
                    
                    result = cursor.fetchall()

                # print(result)

                    cursor.close()
                    connection.close()
                    return result
            except sql.InterfaceError as e:
                logging.error(
                    "Interface Error at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                )
                cursor.close()
                connection.close()
            except RequestError as e:
                logging.error(
                    "RequestError at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                    )
                attempt += 1
                if attempt < retries:
                    logging.info(f"Retrying {attempt}/{retries} in 10 seconds...")
                    time.sleep(10) 
                    retries+=1
            except RequestException as e:
                logging.error(
                    "RequestException at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                    )
                attempt += 1
                if attempt < retries:
                    logging.info(f"Retrying {attempt}/{retries} in 10 seconds...")
                    time.sleep(10) 
                    retries+=1 
            except sql.DatabaseError as e:
                logging.error(
                    "Database Error at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                )
                attempt += 1
                if attempt < retries:
                    logging.info(f"Retrying {attempt}/{retries} in 10 seconds...")
                    time.sleep(10) 
                    retries+=1 
            except (Timeout, ConnectionError) as e:
                logging.error(
                    "Timeout,ConnectionError at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                )
                attempt += 1
                if attempt < retries:
                    logging.info(f"Retrying {attempt}/{retries} in 10 seconds...")
                    time.sleep(10) 
                    retries+=1 
            except sql.OperationalError as e:
                logging.error(
                    "Operational Error at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                )
                cursor.close()
                connection.close()
            except Exception as e:
                cursor.close()
                connection.close()
                logging.error(
                    "Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e
                )
            except RuntimeError as e:
                cursor.close()
                connection.close()
                logging.error("Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e)
            except SyntaxError as e:
                cursor.close()
                connection.close()
                logging.error("Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e)
            finally:
                if cursor:
                    cursor.close()
                if connection:
                    connection.close()
    
    # @retry(wait=wait_exponential(min=2, max=10), stop=stop_after_attempt(5))
    def poll_SilverTableJobDone(
        self, request_id, user_id, tile_count, max_retries, delay=10
    ):
        
        # # Testing existing data
        # user_id = 'cnu4'
        # request_id = '7fe1cd27'
        try:
        
            retries = 0
            connection = None
            cursor = None
            while retries < max_retries:
                try:
                    # if connection is None:
                    #      time.sleep(60)
                        #  continue
                    # if not connection.open:
                    connection = sql.connect(
                       server_hostname="adb-1881246389460182.2.azuredatabricks.net",
                       http_path="/sql/1.0/warehouses/8605a48953a7f210",
                       access_token="dapicb010df06931117a00ccc96cab0abdf0-3",
                       connection_timeout= 600,  # Timeout in seconds
                       socket_timeout=600,
                       retry_config={
                       "min_retry_delay": 1.0,
                       "max_retry_delay": 60.0,
                       "max_attempts": max_retries,
                       "retry_duration": 900.0,
                       "default_retry_delay": 5.0,
                       }
                       )
                    cursor = connection.cursor()
                    query = "SELECT count(bboxes) from edav_dev_csels.towerscout.test_image_silver WHERE user_id = ? AND request_id = ?"
                    # cursor.execute(
                    #     "SELECT count(bboxes) from edav_dev_csels.towerscout.test_image_silver WHERE user_id = '"
                    #     + user_id
                    #     + "' AND request_id = '"
                    #     + request_id
                    #     + "'"
                    # )
                    cursor.execute(query, (user_id, request_id))
                    print(
                        f"SELECT count(bboxes) from edav_dev_csels.towerscout.test_image_silver WHERE user_id = '"
                        + user_id
                        + "' AND request_id = '"
                        + request_id
                        + "'"
                    )
                    logging.info("SELECT count(bboxes) from edav_dev_csels.towerscout.test_image_silver WHERE user_id = '"
                        + user_id
                        + "' AND request_id = '"
                        + request_id
                        + "'")
                    resultcount = cursor.fetchone()
                    print("result count:", resultcount)
                    logging.info(f"result count:{resultcount}")
                    # print("result count[0]:", resultcount[0])
                    # logging.info(f"result count[0]:{resultcount[0]}")
                    # print("tile_count:", tile_count)
                    # logging.info(f"tile_count:{tile_count}")
                    if resultcount[0] >= tile_count:
                        cursor.close()
                        connection.close()
                        print("result count:", resultcount)
                        return True
                    else:
                        retries += 1
                        logging.info(f"Number of retries: {retries}")
                        if retries == max_retries:
                            max_retries +=1
                        cursor.close()
                        connection.close()
                            # raise Timeout("Forcing timeout error")
                        time.sleep(delay)
                except RequestError as e:
                    print("RequestError silvertablejobdone. Retrying...")
                    cursor.close()
                    connection.close()
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                
                except Exception as e:
                    print(f"Exception silvertablejobdone. Retrying...{e}")
                    cursor.close()
                    connection.close()
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                except (Timeout,ConnectionError) as e:
                    print("Timeout,ConnectionError occurred. Retrying...")
                    cursor.close()
                    connection.close()
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                except SyntaxError as e:
                    logging.info(f"result count:{resultcount}")
                    logging.error("SyntaxError at %s","ts_readdetections.py(poll_SilverTableJobDone)",
                    exc_info=e)
                    cursor.close()
                    connection.close()
                    time.sleep(delay)
                except RequestException as e:
                    print("RequestException silvertablejobdone. Retrying...")
                    cursor.close()
                    connection.close()
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                finally:
                    if cursor:
                        cursor.close()
                    if connection:
                        connection.close()
        except sql.InterfaceError as e:
            logging.error(
                "Interface Error at %s",
                "ts_readdetections.py(poll_SilverTableJobDone)",
                exc_info=e
            )
            cursor.close()
            connection.close()
        except sql.DatabaseError as e:
            logging.error(
                "Database Error at %s",
                "ts_readdetections.py(poll_SilverTableJobDone)",
                exc_info=e
            )
            cursor.close()
            connection.close()
        except sql.OperationalError as e:
            logging.error(
                "Operational Error at %s",
                "ts_readdetections.py(poll_SilverTableJobDone)",
                exc_info=e
            )
            cursor.close()
            connection.close()
        except Exception as e:
            cursor.close()
            connection.close()
            logging.error(
                "Error at %s",
                "ts_readdetections.py(poll_SilverTableJobDone)",
                exc_info=e
            )
            cursor.close()
            connection.close()
        except RuntimeError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_SilverTableJobDone)", exc_info=e)
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_SilverTableJobDone)", exc_info=e)
        finally:
                if cursor:
                    cursor.close()
                if connection:
                    connection.close()
      

    def get_bboxesfortiles(self, tiles, events, id, request_id, user_id):
        try:
            tile_count = len(tiles)
            results_raw = self.fetchbboxes(request_id, user_id, tile_count)
            # # Only for localhost - Azure app services
            # for chunk in results_raw:
            #     if chunk:
            #         print(f"result raw chunk: {chunk}")
        
            results = []

            for i in range(0, len(tiles), self.batch_size):
                # make a batch of tiles and detection results
                tile_batch = tiles[i : i + self.batch_size]
                results_raw_batch = results_raw[i : i + self.batch_size]

                for tile, result in zip(tile_batch, results_raw_batch):
                    # record the detections in the tile
                    boxes = []
                    #  get bbox column for each row from the results
                    # result[0] is bboxes, result[1] is tile id
                    bboxarray = result[0]
                
                    tile_results = [
                        {
                            "x1": item["x1"],
                            "y1": item["y1"],
                            "x2": item["x2"],
                            "y2": item["y2"],
                            "conf": item["conf"],
                            "class": int(item["class"]),
                            "class_name": item["class_name"],
                            "secondary": item["secondary"],
                        }
                        for item in bboxarray
                    ]
                    results.append(tile_results)
                    # record the detections in the tile

                    for bbox in tile_results:
                        box = (
                            "0 "
                            + str((bbox["x1"] + bbox["x2"]) / 2)
                            + " "
                            + str((bbox["y1"] + bbox["y2"]) / 2)
                            + " "
                            + str(bbox["x2"] - bbox["x1"])
                            + " "
                            + str(bbox["y2"] - bbox["y1"])
                            + "\n"
                        )
                        boxes.append(box)
                    # tr[1] is tile id
                    tile["detections"] = boxes
                print(f" batch of {len(tile_batch)} processed")

            return results
        except Exception as e:
            logging.error("Error at %s", "get_bboxesfortiles ts_readdetections.py", exc_info=e)
        except RuntimeError as e:
            logging.error("Error at %s", "get_bboxesfortiles ts_readdetections.py", exc_info=e)
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(get_bboxesfortiles)", exc_info=e)
    
    
    def get_bboxesfortilesWithoutPolling(self, tiles, events, id, request_id, user_id):
        try:
            tile_count = len(tiles)
            results_raw = self.fetchbboxesWithoutPolling(request_id, user_id, tile_count)
            # # Only for localhost - Azure app services
            # for chunk in results_raw:
            #     if chunk:
            #         print(f"result raw chunk: {chunk}")
        
            results = []

            for i in range(0, len(tiles), self.batch_size):
                # make a batch of tiles and detection results
                tile_batch = tiles[i : i + self.batch_size]
                results_raw_batch = results_raw[i : i + self.batch_size]

                for tile, result in zip(tile_batch, results_raw_batch):
                    # record the detections in the tile
                    boxes = []
                    #  get bbox column for each row from the results
                    # result[0] is bboxes, result[1] is tile id
                    bboxarray = result[0]
                
                    tile_results = [
                        {
                            "x1": item["x1"],
                            "y1": item["y1"],
                            "x2": item["x2"],
                            "y2": item["y2"],
                            "conf": item["conf"],
                            "class": int(item["class"]),
                            "class_name": item["class_name"],
                            "secondary": item["secondary"],
                        }
                        for item in bboxarray
                    ]
                    results.append(tile_results)
                    # record the detections in the tile

                    for bbox in tile_results:
                        box = (
                            "0 "
                            + str((bbox["x1"] + bbox["x2"]) / 2)
                            + " "
                            + str((bbox["y1"] + bbox["y2"]) / 2)
                            + " "
                            + str(bbox["x2"] - bbox["x1"])
                            + " "
                            + str(bbox["y2"] - bbox["y1"])
                            + "\n"
                        )
                        boxes.append(box)
                    # tr[1] is tile id
                    tile["detections"] = boxes
                    tile["filename"] = result[1]
                    tile["url"] = result[2]
                print(f" batch of {len(tile_batch)} processed")

            return results
        except Exception as e:
            logging.error("Error at %s", "get_bboxesfortiles ts_readdetections.py", exc_info=e)
        except RuntimeError as e:
            logging.error("Error at %s", "get_bboxesfortiles ts_readdetections.py", exc_info=e)
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(get_bboxesfortiles)", exc_info=e)
    

    def get_bboxesfortilesWithoutPollingTest(self, tiles, events, id, request_id, user_id, tiles_count):
        try:
            
            results_all = self.fetchbboxesWithoutPolling(request_id, user_id, tiles_count)
            
            # # Only for localhost - Azure app services
            # for chunk in results_raw:
            #     if chunk:
            #         print(f"result raw chunk: {chunk}")
            
            tiles = []
            results = []

            for i in range(0, len(results_all), self.batch_size):
                # make a batch of tiles and detection results
                # tile_batch = tiles[i : i + self.batch_size]
                # results_raw_batch = results_raw[i : i + self.batch_size]
                results_all_batch = results_all[i : i + self.batch_size]

                # for tile, result in zip(tile_batch, results_raw_batch):
                for resultitem in results_all_batch:
                    # record the detections in the tile
                    boxes = []
                    #  get bbox column for each row from the results
                    # result[0] is bboxes, result[1] is tile id
                    bboxarray = resultitem[0]
                    tile = resultitem[1]
                    tile_results = [
                        {
                            "x1": item["x1"],
                            "y1": item["y1"],
                            "x2": item["x2"],
                            "y2": item["y2"],
                            "conf": item["conf"],
                            "class": int(item["class"]),
                            "class_name": item["class_name"],
                            "secondary": item["secondary"],
                        }
                        for item in bboxarray
                    ]
                    results.append(tile_results)
                    # record the detections in the tile

                    for bbox in tile_results:
                        box = (
                            "0 "
                            + str((bbox["x1"] + bbox["x2"]) / 2)
                            + " "
                            + str((bbox["y1"] + bbox["y2"]) / 2)
                            + " "
                            + str(bbox["x2"] - bbox["x1"])
                            + " "
                            + str(bbox["y2"] - bbox["y1"])
                            + "\n"
                        )
                        boxes.append(box)
                    # tr[1] is tile id
                    tile.update({'detections': boxes})
                    tiles.append(tile)
                    print(f"tile['filename']: {tile}")
                print(f" batch of {len(results_all_batch)} processed")

            return results, tiles
        except Exception as e:
            logging.error("Error at %s", "get_bboxesfortiles ts_readdetections.py", exc_info=e)
        except RuntimeError as e:
            logging.error("Error at %s", "get_bboxesfortiles ts_readdetections.py", exc_info=e)
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(get_bboxesfortiles)", exc_info=e)
