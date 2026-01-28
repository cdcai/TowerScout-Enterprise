from databricks import sql
from requests.exceptions import Timeout, RequestException
import os, time, logging, requests, ts_secrets, json, asyncio, datetime
from databricks.sql.exc import RequestError
from numpy.random import choice
from azure.core.exceptions import ClientAuthenticationError
config_dir = os.path.join(os.getcwd().replace("webapp", ""), "webapp")

class Config:
    DATABRICKS_INSTANCE = ""
    PERSONAL_ACCESS_TOKEN = ts_secrets.getSecret('DB-PERSONAL-ACCESS-TOKEN')
    WAREHOUSE_ID = ""
    SQL_STATEMENTS_ENDPOINT = ""
    CATALOG = ""
    SCHEMA = ""
    TABLE = ""
    GOLD_TABLE = ""
    job_id = ""
class SilverTable:
    

                    
    def __init__(self):
        self.batch_size = 100
        
        if 'WEBSITE_SITE_NAME' in os.environ:
          data = json.loads(os.getenv('dbinstance'))
          Config.job_id = os.getenv('Inference_JOBID')
        else:     
          dbinstancefile = config_dir + "/config.dbinstance.json"
          jobIDFile = config_dir + "/config.dbjobid.json"
          with open(dbinstancefile, "r") as file:
              data = json.load(file)
          with open(jobIDFile, "r") as file:
              jobIDdata = json.load(file)
              Config.job_id = jobIDdata["Inference_JOBID"]
        
        Config.DATABRICKS_INSTANCE = data["DATABRICKS_INSTANCE"]
        Config.WAREHOUSE_ID = data["WAREHOUSE_ID"]
        Config.SCHEMA = data["SCHEMA"]
        Config.CATALOG = data["CATALOG"]
        Config.TABLE = data["TABLE"]
        Config.SQL_STATEMENTS_ENDPOINT = f'https://{Config.DATABRICKS_INSTANCE}/api/2.0/sql/statements'

      # @retry(wait=wait_exponential(min=2, max=10), stop=stop_after_attempt(3))
    def fetchbboxes(self, request_id, user_id, tile_count):
        logging.info("Startinng ts_readdetections.py(fetchbboxes)")
        attempt = 0
        retries = 3
        while attempt < retries:
            try:
                max_retries = tile_count * 2
                jobdone = self.poll_SilverTableJobDone(
                    request_id, user_id, tile_count, max_retries
                )
            
                if jobdone:
                   
                    sql_query  = "SELECT bboxes from edav_prd_csels.towerscout.test_image_silver WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'"
                    # Set the payload for the request (include the query)
                    data = {
                        'statement': sql_query,
                        'warehouse_id': Config.WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
                    }
                    # Set the headers with the personal access token for authentication
                    headers = {
                        'Authorization': f'Bearer {Config.PERSONAL_ACCESS_TOKEN}',
                        'Content-Type': 'application/json',
                    }
            

                    response = requests.post(Config.SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
                    result_data = response.json()
                    print("Query result:", result_data)
                    result_data = result_data['result']['data_array']

                    print("Query result:", result_data)
                    return(result_data)

                
            except sql.InterfaceError as e:
                logging.error(
                    "Interface Error at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                )
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
               
            except Exception as e:
               logging.error(
                    "Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e
                )
            except RuntimeError as e:
               logging.error("Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e)
            except SyntaxError as e:
               logging.error("Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e)
            
    # Using this one
    def fetchbboxesWithoutPolling(self, request_id, user_id, tiles_count):
        logging.info("Startinng ts_readdetections.py(fetchbboxes)")
        attempt = 0
        retries = 3
        while attempt < retries:
            try:
                max_retries = tiles_count * 2
                jobdone = True
                # jobdone = poll_table_for_record_count(
                # request_id, user_id, tile_count, max_retries
                # )
                if jobdone:
                    
                    sql_query  = (f"SELECT bboxes, "
                    "concat(uuid,'.jpeg') AS filename, "
                    "image_path AS url, "
                    "uuid, image_hash "
                    "FROM " + Config.CATALOG + "." + Config.SCHEMA + "." + Config.TABLE + " where user_id = '" + user_id + "' AND request_id = '" + request_id + "' "
                    "ORDER BY image_id")
                    print(f"sql_query: {sql_query}")
                    
                     # Set the payload for the request (include the query)
                    data = {
                        'statement': sql_query,
                        'warehouse_id': Config.WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
                        'output_format': 'json'
                    }
                    # Set the headers with the personal access token for authentication
                    headers = {
                        'Authorization': f'Bearer {Config.PERSONAL_ACCESS_TOKEN}',
                        'Content-Type': 'application/json',
                    }
                    print("Warehouse_ID:", Config.WAREHOUSE_ID)
                    response = requests.post(Config.SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
                    result_data = response.json()
                    print("Query result:", result_data)
                    result_data = result_data['result']['data_array']

                    print("Query result:", result_data)
                    return(result_data)
            except sql.InterfaceError as e:
                logging.error(
                    "Interface Error at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                )
                
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
                
            except Exception as e:
                logging.error(
                    "Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e
                )
            except RuntimeError as e:
                logging.error("Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e)
            except SyntaxError as e:
                logging.error("Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e)
   
    # @retry(wait=wait_exponential(min=2, max=10), stop=stop_after_attempt(5))
    def poll_SilverTableJobDone(
        self, request_id, user_id, tile_count, max_retries, delay=10
    ):
        
        try:
        
            retries = 0
            
            sql_query  = (f"SELECT count(bboxes) "
                          "FROM " + Config.CATALOG + "." + Config.SCHEMA + "." + Config.TABLE + "  WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'")
            # Set the payload for the request (include the query)
            data = {
            'statement': sql_query,
            'warehouse_id': Config.WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
}
            # Set the headers with the personal access token for authentication
            headers = {
            'Authorization': f'Bearer {Config.PERSONAL_ACCESS_TOKEN}',
            'Content-Type': 'application/json',
        }
            while retries < max_retries:
                try:
                    response = requests.post(Config.SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
                    print(f"SELECT count(bboxes) from " + Config.CATALOG + "." + Config.SCHEMA + "." + Config.TABLE + " WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'")
                    print(response.status_code)
                    if response.status_code == 200:
                        result_data = response.json()
                        resultcount = result_data['result']['data_array'][0][0]
                        if (int(resultcount) >= tile_count):
                            
                            print("result count:", resultcount)
                            return True
                        else:
                            retries += 1
                            logging.info(f"Number of retries: {retries}")
                            if retries == max_retries:
                                max_retries +=1
                            
                            # raise Timeout("Forcing timeout error")
                            time.sleep(delay)
                    else:
                        retries += 1
                        logging.info(f"Number of retries: {retries}")
                        if retries == max_retries:
                            max_retries +=1
                        time.sleep(delay)
                   
                    
                except RequestError as e:
                    print("RequestError silvertablejobdone. Retrying...")
                   
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                
                except Exception as e:
                    print(f"Exception silvertablejobdone. Retrying...{e}")
                    
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                except (Timeout,ConnectionError) as e:
                    print("Timeout,ConnectionError occurred. Retrying...")
                    
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                except SyntaxError as e:
                    logging.info(f"result count:{resultcount}")
                    logging.error("SyntaxError at %s","ts_readdetections.py(poll_SilverTableJobDone)",
                    exc_info=e)
                    time.sleep(delay)
                except RequestException as e:
                    print("RequestException silvertablejobdone. Retrying...")
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                except ClientAuthenticationError as e:
                    logging.error("ClientAuthenticationError at %s", "getazmaptransactions towerscout.py", exc_info=e)
                    # Optional: check for expired-secret specifically
                    if "AADSTS7000222" in str(e):
                        raise RuntimeError(
                            "Azure App Service AD client secret has expired. Rotate the secret and update Key Vault."
                        ) from e
                    else:
                        raise
                finally:
                    print("Finally. Retrying sub process...")
                        
        except sql.InterfaceError as e:
            logging.error(
                "Interface Error at %s",
                "ts_readdetections.py(poll_SilverTableJobDone)",
                exc_info=e
            )
            
        except sql.DatabaseError as e:
            logging.error(
                "Database Error at %s",
                "ts_readdetections.py(poll_SilverTableJobDone)",
                exc_info=e
            )
            
        except sql.OperationalError as e:
            logging.error(
                "Operational Error at %s",
                "ts_readdetections.py(poll_SilverTableJobDone)",
                exc_info=e
            )
            
        except Exception as e:
           
            logging.error(
                "Error at %s",
                "ts_readdetections.py(poll_SilverTableJobDone)",
                exc_info=e
            )
           
        except RuntimeError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_SilverTableJobDone)", exc_info=e)
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_SilverTableJobDone)", exc_info=e)
        finally:
            print("Finally. Retrying...")
    
    def poll_SilverTableJobDoneWithLogs(
        self, request_id, user_id, tile_count, max_retries, delay=20, trigger_time_ms=None, session_key=None, exit_events = None
        ):
        
        try:
            if trigger_time_ms is None:
                trigger_time_ms = int(time.time() * 1000)
            # run_id = discover_run_id(Config.job_id, request_id,trigger_time_ms)
            start_time = trigger_time_ms
            retries = 0
            
            totalTilesProcessed = 0
            sql_query  = "SELECT count(bboxes) from " + Config.CATALOG + "." + Config.SCHEMA + "."  + Config.TABLE + " WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'"
            # Set the payload for the request (include the query)
            data = {
            'statement': sql_query,
            'warehouse_id': Config.WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
        }
            # Set the headers with the personal access token for authentication
            headers = {
            'Authorization': f'Bearer {Config.PERSONAL_ACCESS_TOKEN}',
            'Content-Type': 'application/json',
        }
            while True:
                # Check for client abort
                # if session_key and exit_events.query(session_key):
                if session_key and exit_events.query(session_key):
                    logging.info(f"Client aborted polling for session {session_key}")
                    exit_events.free(session_key)
                    yield f"event: error\ndata: Client aborted request\n\n"
                    return

                # Check absolute timeout (10 minutes)
                elapsed_ms = int(time.time() * 1000) - start_time
                if elapsed_ms > 10 * 60 * 1000:  # 10 minutes
                    yield f"event: error\ndata: Polling timed out after 10 minutes\n\n"
                    exit_events.free(session_key)
                    return

                # Build SQL query to count processed tiles
                # job_failed, jobrun_details = get_jobstatus(run_id)
                # if job_failed is True:
                #     yield f"event: error\ndata: Databricks job failed: {json.dumps(jobrun_details)}\n\n"
                #     return
                try:
                    
                    response = requests.post(Config.SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
                    print(f"SELECT count(bboxes) from " + Config.CATALOG + "." + Config.SCHEMA + "."  + Config.TABLE + " WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'")
                    print(response.status_code)
                    response.raise_for_status()
                    if response.status_code == 200:
                        result_data = response.json()
                        newTilesProcessCount = result_data['result']['data_array'][0][0]
                        
                        if (int(newTilesProcessCount) >= tile_count):
                            
                            print("result count:", newTilesProcessCount)
                            final_result = {"status": "Completed", "message": f"Task completed successfully. {newTilesProcessCount} Tiles out of {str(tile_count)} have been processed."}
                            # yield f"data: {jsonify(final_result).data.decode('utf-8')}\n\n"
                            yield f"event: done\ndata: Task completed successfully. All {str(tile_count)} tiles have been processed.\n\n"
                            exit_events.free(session_key)
                            return True
                        else:
                            retries += 1
                            logging.info(f"Number of retries: {retries}")
                            if retries == max_retries:
                                max_retries +=1
                            if (int(newTilesProcessCount) > totalTilesProcessed):
                                logging.info(f"{newTilesProcessCount} Tiles out of {str(tile_count)} have been processed.")
                                # final_result = {"status": "Processing", "message": f"{newTilesProcessCount} Tiles out of {str(tile_count)} have been processed."}
                                yield f"data: {newTilesProcessCount} Tiles out of {str(tile_count)} have been processed.\n\n"
                                # yield f"{newTilesProcessCount} Tiles out of {str(tile_count)} have been processed."
                                print(f"{newTilesProcessCount} Tiles out of {str(tile_count)} have been processed.")
                                totalTilesProcessed = int(newTilesProcessCount)
                            # raise Timeout("Forcing timeout error")
                            time.sleep(delay)
                    elif response.status_code == 400:
                        try:
                            error_data = response.json()
                            logging.error("400 error: %s", "ts_readdetections.py(poll_SilverTableJobDone)")
                            # Common patterns
                            field = error_data.get("field")     
                            message = error_data.get("message")
                            errors = error_data.get("errors")

                            print("Invalid field:", field or errors or message)
                        except ValueError as e:
                            logging.error("400 error with non-JSON body: %s", "ts_readdetections.py(poll_SilverTableJobDone)", exc_info=e)
                            raise    

                        
                    else:
                        retries += 1
                        logging.info(f"Number of retries: {retries}")
                        if retries == max_retries:
                            max_retries +=1
                                                      
                        time.sleep(delay)
                except KeyError as e:
                    missing_key = e.args[0]
                    logging.error(f"Error at %s Missing key: {missing_key}", "ts_readdetections.py(poll_SilverTableJobDone)", exc_info=e)
                    print(f"Missing key: {missing_key}")
                    logging.error(f"Missing key in API response: {missing_key}", exc_info=e)
                    yield f"event: error\ndata: Missing key in response: {missing_key}\n\n"
                    exit_events.free(session_key)
                    raise     
                except RuntimeError as e:
                    logging.error("Error at %s", "ts_readdetections.py(poll_SilverTableJobDone)", exc_info=e)
                    exit_events.free(session_key)
                    raise   
                except RequestError as e:
                    print("RequestError silvertablejobdone. Retrying...")

                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                
                except (Timeout,ConnectionError) as e:
                    print("Timeout,ConnectionError occurred. Retrying...")
                    
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                except SyntaxError as e:
                    logging.info(f"result count:{newTilesProcessCount}")
                    logging.error("SyntaxError at %s","ts_readdetections.py(poll_SilverTableJobDone)",
                    exc_info=e)
                    exit_events.free(session_key)
                    raise
                    # time.sleep(delay)
                    
                except RequestException as e:
                    print("RequestException silvertablejobdone. Retrying...")
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                except Exception as e:
                    print(f"Exception silvertablejobdone. Retrying...{e}")
                    
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                
                finally:
                    print("Finally. Retrying sub process...")
                        
        except sql.InterfaceError as e:
            logging.error(
                "Interface Error at %s",
                "ts_readdetections.py(poll_SilverTableJobDone)",
                exc_info=e
            )
            raise
        except sql.DatabaseError as e:
            logging.error(
                "Database Error at %s",
                "ts_readdetections.py(poll_SilverTableJobDone)",
                exc_info=e
            )
            raise
        except sql.OperationalError as e:
            logging.error(
                "Operational Error at %s",
                "ts_readdetections.py(poll_SilverTableJobDone)",
                exc_info=e
            )
            raise
        
        except RuntimeError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_SilverTableJobDone)", exc_info=e)
            raise
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_SilverTableJobDone)", exc_info=e)
            raise
        except Exception as e:
           
            logging.error(
                "Error at %s",
                "ts_readdetections.py(poll_SilverTableJobDone)",
                exc_info=e
            )
            raise
        finally:
            print("Finally. Retrying...")
            if session_key:
                exit_events.free(session_key)
     
    def get_bboxesfortiles(self, tiles, events, id, request_id, user_id):
        try:
            tile_count = len(tiles)
            results_raw = self.fetchbboxes(request_id, user_id, tile_count)
            
        
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
                    bboxarray = json.loads(result[0])
                
                    tile_results = [
                        {
                        'x1': float(item['x1']),
                        'y1':float(item['y1']),
                        'x2':float(item['x2']),
                        'y2':float(item['y2']),
                        'conf':float(item['conf']),
                        'class':int(item['class']),
                        'class_name':item['class_name'],
                        'secondary':float(item['secondary']),
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
                    bboxarray = json.loads(result[0])
                
                    tile_results = [
                        {
                        'x1': float(item['x1']),
                        'y1':float(item['y1']),
                        'x2':float(item['x2']),
                        'y2':float(item['y2']),
                        'conf':float(item['conf']),
                        'class':int(item['class']),
                        'class_name':item['class_name'],
                        'secondary':float(item['secondary']),
                        'uuid': result[3],
                        'image_hash': result[4],
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
                    tile["uuid"] = result[3]
                    tile["image_hash"] = result[4]
                print(f" batch of {len(tile_batch)} processed")

            return results
        except Exception as e:
            logging.error("Error at %s", "get_bboxesfortiles ts_readdetections.py", exc_info=e)
        except RuntimeError as e:
            logging.error("Error at %s", "get_bboxesfortiles ts_readdetections.py", exc_info=e)
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(get_bboxesfortiles)", exc_info=e)
    
    # Working
    # Function to dynamically construct the SQL query and execute the merge
    def PerformSilverToGoldmerge(self, catalog, schema, silver_table, gold_table, silver_detections, user_id, request_id):
        try:

            # Generate SQL-compatible values string for the JSON data
            values_str = ', '.join([f"('{row['uuid']}', '{row['image_hash']}', '{json.dumps(row['bboxes'])}', '{choice(a=['train', 'val', 'test'], p=[0.8, 0.1, 0.1])}')"
                         for row in silver_detections])
            print("values_Str:", values_str)
            
         # Define the raw SQL query (MERGE query)
            merge_query = f"""
            WITH silverdetections AS (
                SELECT * FROM (VALUES {values_str}) AS silverdetections(uuid, image_hash, bboxes, split_label)
            )
            MERGE INTO {gold_table} AS target
            USING (
        SELECT source.*, silverdetections.image_hash AS sd_image_hash, 
        from_json(silverdetections.bboxes, 'ARRAY<STRUCT<x2: FLOAT, x1: FLOAT, y2: FLOAT, y1: FLOAT, class_name: STRING, conf: FLOAT, secondary: FLOAT, class: INT>>') AS sd_bboxes,
        silverdetections.split_label AS sd_split_label
        FROM {silver_table} AS source
        LEFT JOIN silverdetections
        ON source.image_hash = silverdetections.image_hash
        WHERE source.user_id = '{user_id}' AND source.request_id = '{request_id}'
        ) AS source
            ON 
            target.image_hash = source.image_hash
            WHEN MATCHED THEN
                UPDATE SET
                target.user_id = source.user_id,
                target.request_id = source.request_id,
                target.uuid = source.uuid,
                target.reviewed_time = CURRENT_TIMESTAMP(),
                target.bboxes = source.sd_bboxes,
                target.image_hash = source.image_hash,
                target.image_path = source.image_path,
                target.split_label = source.sd_split_label
            WHEN NOT MATCHED THEN
                INSERT (user_id, request_id, uuid, reviewed_time, bboxes, image_hash, image_path, split_label)
                VALUES (source.user_id, source.request_id, source.uuid, CURRENT_TIMESTAMP(), source.sd_bboxes, source.image_hash, source.image_path, source.sd_split_label);
            """


            print(f"merge_query: {merge_query}")
        
            response_data = self.execute_sql_query(merge_query, catalog=catalog, schema=schema)
            print(f"execute_sql_query response: {response_data}")
            return response_data
            
            
        except Exception as ex:
            logging.error("Error at %s", "ts_readdetections.py(PerformSilverToGoldmerge)", exc_info=ex)
        
    # Working
    def execute_sql_query(self, sql_query, catalog, schema):
        try:
           
            headers = {
                "Authorization": f"Bearer {Config.PERSONAL_ACCESS_TOKEN}",
                "Content-Type": "application/json"
            }
    
            body = {
                "warehouse_id": Config.WAREHOUSE_ID,
                "statement": sql_query,
                "catalog": catalog,
                "schema": schema
            }
    
            response = requests.post(Config.SQL_STATEMENTS_ENDPOINT, json=body, headers=headers)
            statement_id = ""
            logging.info(f"executed merge query: {response.json()}")
            if response.status_code == 200:

                statement_id = response.json().get('statement_id')
                return statement_id
           
        except sql.InterfaceError as e:
            logging.error(
                    "Interface Error at %s",
                    "ts_readdetections.py(execute_sql_queryes)",
                    exc_info=e
                )
            
        except RequestError as e:
            logging.error(
                    "RequestError at %s",
                    "ts_readdetections.py(execute_sql_queryes)",
                    exc_info=e
                    )
           
        except RequestException as e:
            logging.error(
                    "RequestException at %s",
                    "ts_readdetections.py(execute_sql_queryes)",
                    exc_info=e
                    )
            
        except sql.DatabaseError as e:
            logging.error(
                    "Database Error at %s",
                    "ts_readdetections.py(execute_sql_queryes)",
                    exc_info=e
                )
            
        except (Timeout, ConnectionError) as e:
            logging.error(
                    "Timeout,ConnectionError at %s",
                    "ts_readdetections.py(execute_sql_queryes)",
                    exc_info=e
                )
            
        except sql.OperationalError as e:
            logging.error(
                    "Operational Error at %s",
                    "ts_readdetections.py(execute_sql_queryes)",
                    exc_info=e
                )
            
        except AttributeError as e:
            logging.error(
                    "AttributeError Error at %s", "ts_readdetections.py(execute_sql_queryes)", exc_info=e
                )  
            
        except Exception as e:
            logging.error(
                    "Error at %s", "ts_readdetections.py(execute_sql_queryes)", exc_info=e
                )
            
        except RuntimeError as e:
            logging.error("Error at %s", "ts_readdetections.py(execute_sql_queryes)", exc_info=e)
            
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(execute_sql_query)", exc_info=e)
        finally:
            return statement_id   
      
 
    def poll_query_status(self, statement_id, delay=5):
        try:
            while True:
                try:
                    success, error_message = self.check_query_status(statement_id)
                    if success:
                        print("Query succeeded.")
                        return True
                    elif success is False:
                        print(f"Query failed: {error_message}")
                        time.sleep(delay)  # Wait before retrying
                        return False
                    else:
                        print("Query still running... Retrying in 5 seconds.")
                        time.sleep(delay)
                        return False

                except RequestError as e:
                        print("RequestError poll_query_Status. Retrying...")
                   
                        time.sleep(delay)  # Wait before retrying
                        return False
                except Exception as e:
                    print(f"Exception poll_query_Status. Retrying...{e}")
               
                    time.sleep(delay)  # Wait before retrying
                    return False
                except (Timeout,ConnectionError) as e:
                    print("Timeout,ConnectionError occurred poll_query_Status. Retrying...")
             
                    time.sleep(delay)  # Wait before retrying
                    return False
                except SyntaxError as e:
                    logging.error("SyntaxError at %s","ts_readdetections.py(poll_query_Status)",
                    exc_info=e)
                    time.sleep(delay)
                    return False
                except RequestException as e:
                    print("RequestException poll_query_Status. Retrying...")
               
                    time.sleep(delay)  # Wait before retrying
                    return False
        except sql.InterfaceError as e:
            logging.error(
                "Interface Error at %s",
                "ts_readdetections.py(poll_query_status)",
                exc_info=e
            )
            
        except sql.DatabaseError as e:
            logging.error(
                "Database Error at %s",
                "ts_readdetections.py(poll_query_status)",
                exc_info=e
            )
            
        except sql.OperationalError as e:
            logging.error(
                "Operational Error at %s",
                "ts_readdetections.py(poll_query_status)",
                exc_info=e
            )
            
        except Exception as e:
           
            logging.error(
                "Error at %s",
                "ts_readdetections.py(poll_query_status)",
                exc_info=e
            )
           
        except RuntimeError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_query_status)", exc_info=e)
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_query_status)", exc_info=e)
        finally:
            print("Finally. Retrying...")


    def check_query_status(self, statement_id):
        url = f'https://{Config.DATABRICKS_INSTANCE}/api/2.0/sql/statements/{statement_id}'
        headers = {
            "Authorization": f"Bearer {Config.PERSONAL_ACCESS_TOKEN}"
        }
     
        try:
            response = requests.get(url, headers=headers)
            error_message = ""
            if response.status_code == 200:
                status = response.json()
                execution_state = status.get('status')['state']  # "RUNNING", "SUCCEEDED", "FAILED", etc.
                error_message = status.get('error_message', None)
                print(f"Query execution state: {execution_state}")
                if execution_state == "SUCCEEDED":
                    print("Query completed successfully.")
                    return True, None
                elif execution_state == "FAILED":
                    print(f"Query failed: {error_message}")
                    return False, error_message
                else:
                    print("Query is still running...")
                    return None, None  # Query is still running, continue polling
                
            else:
                print(f"Error fetching query status: {response.text}")
                return False, error_message
        except sql.InterfaceError as e:
            logging.error(
                    "Interface Error at %s",
                    "ts_readdetections.py(check_query_status)",
                    exc_info=e
                )
            return False, None
        except RequestError as e:
            logging.error(
                    "RequestError at %s",
                    "ts_readdetections.py(check_query_status)",
                    exc_info=e
                    )
            return False, None
        except RequestException as e:
            logging.error(
                    "RequestException at %s",
                    "ts_readdetections.py(check_query_status)",
                    exc_info=e
                    )
            return False, None
        except sql.DatabaseError as e:
            logging.error(
                    "Database Error at %s",
                    "ts_readdetections.py(check_query_status)",
                    exc_info=e
                )
            return False, None
        except (Timeout, ConnectionError) as e:
            logging.error(
                    "Timeout,ConnectionError at %s",
                    "ts_readdetections.py(check_query_status)",
                    exc_info=e
                )
            return False, None
        except sql.OperationalError as e:
            logging.error(
                    "Operational Error at %s",
                    "ts_readdetections.py(check_query_status)",
                    exc_info=e
                )
            return False, None  
        except Exception as e:
            logging.error(
                    "Error at %s", "ts_readdetections.py(check_query_status)", exc_info=e
                )
            return False, None
        except RuntimeError as e:
            logging.error("Error at %s", "ts_readdetections.py(check_query_status)", exc_info=e)
            return False, None
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(check_query_status)", exc_info=e)
            return False, None

class GoldTable:
    
                     
    def __init__(self):
        self.batch_size = 100
        if 'WEBSITE_SITE_NAME' in os.environ:
          data = json.loads(os.getenv('dbinstance'))
          Config.job_id = os.getenv('Inference_JOBID')
        else:     
          dbinstancefile = config_dir + "/config.dbinstance.json"
          jobIDFile = config_dir + "/config.dbjobid.json"
          with open(dbinstancefile, "r") as file:
              data = json.load(file)
          with open(jobIDFile, "r") as file:
              jobIDdata = json.load(jobIDFile)
              Config.job_id = jobIDdata('Inference_JOBID')
       
        Config.DATABRICKS_INSTANCE = data["DATABRICKS_INSTANCE"]
        Config.WAREHOUSE_ID = data["WAREHOUSE_ID"]
        Config.SCHEMA = data["SCHEMA"]
        Config.CATALOG = data["CATALOG"]
        Config.TABLE = data["TABLE"]
        Config.GOLD_TABLE = data["GOLD_TABLE"]
        Config.SQL_STATEMENTS_ENDPOINT = f'https://{Config.DATABRICKS_INSTANCE}/api/2.0/sql/statements'

      # @retry(wait=wait_exponential(min=2, max=10), stop=stop_after_attempt(3))
    def fetchbboxes(self, request_id, user_id, tile_count):
        logging.info("Startinng ts_readdetections.py(fetchbboxes)")
        attempt = 0
        retries = 3
        while attempt < retries:
            try:
                max_retries = tile_count * 2
                jobdone = self.poll_GoldTableJobDone(
                    request_id, user_id, tile_count, max_retries
                )
            
                if jobdone:
                    
                    sql_query  = "SELECT bboxes from edav_prd_csels.towerscout.test_image_gold WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'"
                    # Set the payload for the request (include the query)
                    data = {
                        'statement': sql_query,
                        'warehouse_id': Config.WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
                    }
                    # Set the headers with the personal access token for authentication
                    headers = {
                        'Authorization': f'Bearer {Config.PERSONAL_ACCESS_TOKEN}',
                        'Content-Type': 'application/json',
                    }
            

                    response = requests.post(Config.SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
                    result_data = response.json()
                    print("Query result:", result_data)
                    result_data = result_data['result']['data_array']

                    print("Query result:", result_data)
                    return(result_data)

                
            except sql.InterfaceError as e:
                logging.error(
                    "Interface Error at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                )
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
               
            except Exception as e:
               logging.error(
                    "Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e
                )
            except RuntimeError as e:
               logging.error("Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e)
            except SyntaxError as e:
               logging.error("Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e)
            
    # Using this one
    def fetchbboxesWithoutPolling(self, request_id, user_id, tiles_count):
        logging.info("Startinng ts_readdetections.py(fetchbboxes)")
        attempt = 0
        retries = 3
        while attempt < retries:
            try:
                max_retries = tiles_count * 2
                jobdone = True
                # jobdone = poll_table_for_record_count(
                # request_id, user_id, tile_count, max_retries
                # )
                if jobdone:
                   
                    sql_query  = (f"SELECT bboxes, "
                    "concat(uuid,'.jpeg') AS filename, "
                    "image_path AS url, "
                    "uuid, image_hash "
                    "FROM " + Config.CATALOG + "." + Config.SCHEMA + "." + Config.GOLD_TABLE + " where user_id = '" + user_id + "' AND request_id = '" + request_id + "' ")
                    print(f"sql_query: {sql_query}")
                    
                     # Set the payload for the request (include the query)
                    data = {
                        'statement': sql_query,
                        'warehouse_id': Config.WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
                        'output_format': 'json'
                    }
                    # Set the headers with the personal access token for authentication
                    headers = {
                        'Authorization': f'Bearer {Config.PERSONAL_ACCESS_TOKEN}',
                        'Content-Type': 'application/json',
                    }
            
                    response = requests.post(Config.SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
                    result_data = response.json()
                    print("Query result:", result_data)
                    result_data = result_data['result']['data_array']

                    print("Query result:", result_data)
                    return(result_data)
            except sql.InterfaceError as e:
                logging.error(
                    "Interface Error at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                )
                
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
                
            except Exception as e:
                logging.error(
                    "Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e
                )
            except RuntimeError as e:
                logging.error("Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e)
            except SyntaxError as e:
                logging.error("Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e)
  
    # @retry(wait=wait_exponential(min=2, max=10), stop=stop_after_attempt(5))
    def poll_GoldTableJobDone(
        self, request_id, user_id, tile_count, max_retries, delay=10
    ):
        
        
        try:
        
            retries = 0
            
            sql_query  = (f"SELECT count(bboxes) "
                          "FROM " + Config.CATALOG + "." + Config.SCHEMA + "." + Config.TABLE + "  WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'")
# Set the payload for the request (include the query)
            data = {
            'statement': sql_query,
            'warehouse_id': Config.WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
}
        # Set the headers with the personal access token for authentication
            headers = {
            'Authorization': f'Bearer {Config.PERSONAL_ACCESS_TOKEN}',
            'Content-Type': 'application/json',
        }
            while retries < max_retries:
                try:
                    # if connection is None:
                    #      time.sleep(60)
                        #  continue
                    # if not connection.open:
                    response = requests.post(Config.SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
                    print(f"SELECT count(bboxes) from " + Config.CATALOG + "." + Config.SCHEMA + "." + Config.TABLE + " WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'")
                    print(response.status_code)
                    if response.status_code == 200:
                        result_data = response.json()
                        resultcount = result_data['result']['data_array'][0][0]
                        if (int(resultcount) >= tile_count):
                            
                            print("result count:", resultcount)
                            return True
                        else:
                            retries += 1
                            logging.info(f"Number of retries: {retries}")
                            if retries == max_retries:
                                max_retries +=1
                            
                            # raise Timeout("Forcing timeout error")
                            time.sleep(delay)
                    else:
                        retries += 1
                        logging.info(f"Number of retries: {retries}")
                        if retries == max_retries:
                            max_retries +=1
                        time.sleep(delay)
                   
                    
                except RequestError as e:
                    print("RequestError Goldtablejobdone. Retrying...")
                   
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                
                except Exception as e:
                    print(f"Exception Goldtablejobdone. Retrying...{e}")
                    
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                except (Timeout,ConnectionError) as e:
                    print("Timeout,ConnectionError occurred. Retrying...")
                    
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                except SyntaxError as e:
                    logging.info(f"result count:{resultcount}")
                    logging.error("SyntaxError at %s","ts_readdetections.py(poll_GoldTableJobDone)",
                    exc_info=e)
                    time.sleep(delay)
                except RequestException as e:
                    print("RequestException Goldtablejobdone. Retrying...")
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                finally:
                    print("Finally. Retrying sub process...")
                        
        except sql.InterfaceError as e:
            logging.error(
                "Interface Error at %s",
                "ts_readdetections.py(poll_GoldTableJobDone)",
                exc_info=e
            )
            
        except sql.DatabaseError as e:
            logging.error(
                "Database Error at %s",
                "ts_readdetections.py(poll_GoldTableJobDone)",
                exc_info=e
            )
            
        except sql.OperationalError as e:
            logging.error(
                "Operational Error at %s",
                "ts_readdetections.py(poll_GoldTableJobDone)",
                exc_info=e
            )
            
        except Exception as e:
           
            logging.error(
                "Error at %s",
                "ts_readdetections.py(poll_GoldTableJobDone)",
                exc_info=e
            )
           
        except RuntimeError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_GoldTableJobDone)", exc_info=e)
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_GoldTableJobDone)", exc_info=e)
        finally:
            print("Finally. Retrying...")
    
    def poll_GoldTableJobDoneWithLogs(
        self, request_id, user_id, tile_count, max_retries, delay=10
        ):
        print("calling poll_GoldTableJobDoneWithLogs")
        
        try:
        
            retries = 0
            totalTilesProcessed = 0
            sql_query  = "SELECT count(bboxes) from " + Config.CATALOG + "." + Config.SCHEMA + "."  + Config.TABLE + " WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'"
# Set the payload for the request (include the query)
            data = {
            'statement': sql_query,
            'warehouse_id': Config.WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
        }
        # Set the headers with the personal access token for authentication
            headers = {
            'Authorization': f'Bearer {Config.PERSONAL_ACCESS_TOKEN}',
            'Content-Type': 'application/json',
        }
            while retries < max_retries:
                try:
                    # if connection is None:
                    #      time.sleep(60)
                        #  continue
                    # if not connection.open:
                    response = requests.post(Config.SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
                    print(f"SELECT count(bboxes) from " + Config.CATALOG + "." + Config.SCHEMA + "."  + Config.TABLE + " WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'")
                    print(response.status_code)
                    if response.status_code == 200:
                        result_data = response.json()
                        newTilesProcessCount = result_data['result']['data_array'][0][0]
                        
                        if (int(newTilesProcessCount) >= tile_count):
                            
                            print("result count:", newTilesProcessCount)
                            final_result = {"status": "Completed", "message": f"Task completed successfully. {newTilesProcessCount} Tiles out of {str(tile_count)} have been processed."}
                            # yield f"data: {jsonify(final_result).data.decode('utf-8')}\n\n"
                            yield f"event: done\ndata: Task completed successfully. All {str(tile_count)} tiles have been processed.\n\n"
                            return True
                        else:
                            retries += 1
                            logging.info(f"Number of retries: {retries}")
                            if retries == max_retries:
                                max_retries +=1
                            if (int(newTilesProcessCount) > totalTilesProcessed):
                                logging.info(f"{newTilesProcessCount} Tiles out of {str(tile_count)} have been processed.")
                                # final_result = {"status": "Processing", "message": f"{newTilesProcessCount} Tiles out of {str(tile_count)} have been processed."}
                                yield f"data: {newTilesProcessCount} Tiles out of {str(tile_count)} have been processed.\n\n"
                                # yield f"{newTilesProcessCount} Tiles out of {str(tile_count)} have been processed."
                                print(f"{newTilesProcessCount} Tiles out of {str(tile_count)} have been processed.")
                                totalTilesProcessed = int(newTilesProcessCount)
                            # raise Timeout("Forcing timeout error")
                            time.sleep(delay)
                    else:
                        retries += 1
                        logging.info(f"Number of retries: {retries}")
                        if retries == max_retries:
                            max_retries +=1
                        time.sleep(delay)
                   
                    
                except RequestError as e:
                    print("RequestError Goldtablejobdone. Retrying...")
                   
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                
                except Exception as e:
                    print(f"Exception Goldtablejobdone. Retrying...{e}")
                    
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                except (Timeout,ConnectionError) as e:
                    print("Timeout,ConnectionError occurred. Retrying...")
                    
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                except SyntaxError as e:
                    logging.info(f"result count:{newTilesProcessCount}")
                    logging.error("SyntaxError at %s","ts_readdetections.py(poll_GoldTableJobDone)",
                    exc_info=e)
                    time.sleep(delay)
                except RequestException as e:
                    print("RequestException Goldtablejobdone. Retrying...")
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
                finally:
                    print("Finally. Retrying sub process...")
                        
        except sql.InterfaceError as e:
            logging.error(
                "Interface Error at %s",
                "ts_readdetections.py(poll_GoldTableJobDone)",
                exc_info=e
            )
            
        except sql.DatabaseError as e:
            logging.error(
                "Database Error at %s",
                "ts_readdetections.py(poll_GoldTableJobDone)",
                exc_info=e
            )
            
        except sql.OperationalError as e:
            logging.error(
                "Operational Error at %s",
                "ts_readdetections.py(poll_GoldTableJobDone)",
                exc_info=e
            )
            
        except Exception as e:
           
            logging.error(
                "Error at %s",
                "ts_readdetections.py(poll_GoldTableJobDone)",
                exc_info=e
            )
           
        except RuntimeError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_GoldTableJobDone)", exc_info=e)
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_GoldTableJobDone)", exc_info=e)
        finally:
            print("Finally. Retrying...")
                    
      

    def get_bboxesfortiles(self, tiles, events, id, request_id, user_id):
        try:
            tile_count = len(tiles)
            results_raw = self.fetchbboxes(request_id, user_id, tile_count)
           
        
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
                    bboxarray = json.loads(result[0])
                
                    tile_results = [
                        {
                        'x1': float(item['x1']),
                        'y1':float(item['y1']),
                        'x2':float(item['x2']),
                        'y2':float(item['y2']),
                        'conf':float(item['conf']),
                        'class':int(item['class']),
                        'class_name':item['class_name'],
                        'secondary':float(item['secondary']),
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
                    bboxarray = json.loads(result[0])
                
                    tile_results = [
                        {
                        'x1': float(item['x1']),
                        'y1':float(item['y1']),
                        'x2':float(item['x2']),
                        'y2':float(item['y2']),
                        'conf':float(item['conf']),
                        'class':int(item['class']),
                        'class_name':item['class_name'],
                        'secondary':float(item['secondary']),
                        'uuid': result[3],
                        'image_hash': result[4],
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
                    tile["uuid"] = result[3]
                    tile["image_hash"] = result[4]
                print(f" batch of {len(tile_batch)} processed")

            return results
        except Exception as e:
            logging.error("Error at %s", "get_bboxesfortiles ts_readdetections.py", exc_info=e)
        except RuntimeError as e:
            logging.error("Error at %s", "get_bboxesfortiles ts_readdetections.py", exc_info=e)
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(get_bboxesfortiles)", exc_info=e)
    
    # Working
    # Function to dynamically construct the SQL query and execute the merge
    def PerformSilverToGoldmerge(self, catalog, schema, silver_table, gold_table, silver_detections, user_id, request_id):
        try:

            # Generate SQL-compatible values string for the JSON data
            values_str = ', '.join([f"('{row['uuid']}', '{row['image_hash']}', '{json.dumps(row['bboxes'])}', '{choice(a=['train', 'val', 'test'], p=[0.8, 0.1, 0.1])}')"
                         for row in silver_detections])
            print("values_Str:", values_str)
            
         # Define the raw SQL query (MERGE query)
            merge_query = f"""
            WITH silverdetections AS (
                SELECT * FROM (VALUES {values_str}) AS silverdetections(uuid, image_hash, bboxes, split_label)
            )
            MERGE INTO {gold_table} AS target
            USING (
        SELECT source.*, silverdetections.image_hash AS sd_image_hash, 
        from_json(silverdetections.bboxes, 'ARRAY<STRUCT<x2: FLOAT, x1: FLOAT, y2: FLOAT, y1: FLOAT, class_name: STRING, conf: FLOAT, secondary: FLOAT, class: INT>>') AS sd_bboxes,
        silverdetections.split_label AS sd_split_label
        FROM {silver_table} AS source
        LEFT JOIN silverdetections
        ON source.image_hash = silverdetections.image_hash
        WHERE source.user_id = '{user_id}' AND source.request_id = '{request_id}'
        ) AS source
            ON 
            target.image_hash = source.image_hash
            WHEN MATCHED THEN
                UPDATE SET
                target.user_id = source.user_id,
                target.request_id = source.request_id,
                target.uuid = source.uuid,
                target.reviewed_time = CURRENT_TIMESTAMP(),
                target.bboxes = source.sd_bboxes,
                target.image_hash = source.image_hash,
                target.image_path = source.image_path,
                target.split_label = source.sd_split_label
            WHEN NOT MATCHED THEN
                INSERT (user_id, request_id, uuid, reviewed_time, bboxes, image_hash, image_path, split_label)
                VALUES (source.user_id, source.request_id, source.uuid, CURRENT_TIMESTAMP(), source.sd_bboxes, source.image_hash, source.image_path, source.sd_split_label);
            """


            print(f"merge_query: {merge_query}")
        
            response_data = self.execute_sql_query(merge_query, catalog=catalog, schema=schema)
            print(f"execute_sql_query response: {response_data}")
            return response_data
            
            
        except Exception as ex:
            logging.error("Error at %s", "ts_readdetections.py(PerformSilverToGoldmerge)", exc_info=ex)
        
    # Working
    def execute_sql_query(self, sql_query, catalog, schema):
        try:
           
            headers = {
                "Authorization": f"Bearer {Config.PERSONAL_ACCESS_TOKEN}",
                "Content-Type": "application/json"
            }
    
            body = {
                "warehouse_id": Config.WAREHOUSE_ID,
                "statement": sql_query,
                "catalog": catalog,
                "schema": schema
            }
    
            response = requests.post(Config.SQL_STATEMENTS_ENDPOINT, json=body, headers=headers)
            statement_id = ""
            logging.info(f"executed merge query: {response.json()}")
            if response.status_code == 200:

                statement_id = response.json().get('statement_id')
                return statement_id
           
        except sql.InterfaceError as e:
            logging.error(
                    "Interface Error at %s",
                    "ts_readdetections.py(execute_sql_queryes)",
                    exc_info=e
                )
            
        except RequestError as e:
            logging.error(
                    "RequestError at %s",
                    "ts_readdetections.py(execute_sql_queryes)",
                    exc_info=e
                    )
           
        except RequestException as e:
            logging.error(
                    "RequestException at %s",
                    "ts_readdetections.py(execute_sql_queryes)",
                    exc_info=e
                    )
            
        except sql.DatabaseError as e:
            logging.error(
                    "Database Error at %s",
                    "ts_readdetections.py(execute_sql_queryes)",
                    exc_info=e
                )
            
        except (Timeout, ConnectionError) as e:
            logging.error(
                    "Timeout,ConnectionError at %s",
                    "ts_readdetections.py(execute_sql_queryes)",
                    exc_info=e
                )
            
        except sql.OperationalError as e:
            logging.error(
                    "Operational Error at %s",
                    "ts_readdetections.py(execute_sql_queryes)",
                    exc_info=e
                )
            
        except AttributeError as e:
            logging.error(
                    "AttributeError Error at %s", "ts_readdetections.py(execute_sql_queryes)", exc_info=e
                )  
            
        except Exception as e:
            logging.error(
                    "Error at %s", "ts_readdetections.py(execute_sql_queryes)", exc_info=e
                )
            
        except RuntimeError as e:
            logging.error("Error at %s", "ts_readdetections.py(execute_sql_queryes)", exc_info=e)
            
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(execute_sql_query)", exc_info=e)
        finally:
            return statement_id   
      
 
    def poll_query_status(self, statement_id, delay=5):
        try:
            while True:
                try:
                    success, error_message = self.check_query_status(statement_id)
                    if success:
                        print("Query succeeded.")
                        return True
                    elif success is False:
                        print(f"Query failed: {error_message}")
                        time.sleep(delay)  # Wait before retrying
                        return False
                    else:
                        print("Query still running... Retrying in 5 seconds.")
                        time.sleep(delay)
                        return False

                except RequestError as e:
                        print("RequestError poll_query_Status. Retrying...")
                   
                        time.sleep(delay)  # Wait before retrying
                        return False
                except Exception as e:
                    print(f"Exception poll_query_Status. Retrying...{e}")
               
                    time.sleep(delay)  # Wait before retrying
                    return False
                except (Timeout,ConnectionError) as e:
                    print("Timeout,ConnectionError occurred poll_query_Status. Retrying...")
             
                    time.sleep(delay)  # Wait before retrying
                    return False
                except SyntaxError as e:
                    logging.error("SyntaxError at %s","ts_readdetections.py(poll_query_Status)",
                    exc_info=e)
                    time.sleep(delay)
                    return False
                except RequestException as e:
                    print("RequestException poll_query_Status. Retrying...")
               
                    time.sleep(delay)  # Wait before retrying
                    return False
        except sql.InterfaceError as e:
            logging.error(
                "Interface Error at %s",
                "ts_readdetections.py(poll_query_status)",
                exc_info=e
            )
            
        except sql.DatabaseError as e:
            logging.error(
                "Database Error at %s",
                "ts_readdetections.py(poll_query_status)",
                exc_info=e
            )
            
        except sql.OperationalError as e:
            logging.error(
                "Operational Error at %s",
                "ts_readdetections.py(poll_query_status)",
                exc_info=e
            )
            
        except Exception as e:
           
            logging.error(
                "Error at %s",
                "ts_readdetections.py(poll_query_status)",
                exc_info=e
            )
           
        except RuntimeError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_query_status)", exc_info=e)
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_query_status)", exc_info=e)
        finally:
            print("Finally. Retrying...")


    def check_query_status(self, statement_id):
        url = f'https://{Config.DATABRICKS_INSTANCE}/api/2.0/sql/statements/{statement_id}'
        headers = {
            "Authorization": f"Bearer {Config.PERSONAL_ACCESS_TOKEN}"
        }
     
        try:
            response = requests.get(url, headers=headers)
            error_message = ""
            if response.status_code == 200:
                status = response.json()
                execution_state = status.get('status')['state']  # "RUNNING", "SUCCEEDED", "FAILED", etc.
                error_message = status.get('error_message', None)
                print(f"Query execution state: {execution_state}")
                if execution_state == "SUCCEEDED":
                    print("Query completed successfully.")
                    return True, None
                elif execution_state == "FAILED":
                    print(f"Query failed: {error_message}")
                    return False, error_message
                else:
                    print("Query is still running...")
                    return None, None  # Query is still running, continue polling
                
            else:
                print(f"Error fetching query status: {response.text}")
                return False, error_message
        except sql.InterfaceError as e:
            logging.error(
                    "Interface Error at %s",
                    "ts_readdetections.py(check_query_status)",
                    exc_info=e
                )
            return False, None
        except RequestError as e:
            logging.error(
                    "RequestError at %s",
                    "ts_readdetections.py(check_query_status)",
                    exc_info=e
                    )
            return False, None
        except RequestException as e:
            logging.error(
                    "RequestException at %s",
                    "ts_readdetections.py(check_query_status)",
                    exc_info=e
                    )
            return False, None
        except sql.DatabaseError as e:
            logging.error(
                    "Database Error at %s",
                    "ts_readdetections.py(check_query_status)",
                    exc_info=e
                )
            return False, None
        except (Timeout, ConnectionError) as e:
            logging.error(
                    "Timeout,ConnectionError at %s",
                    "ts_readdetections.py(check_query_status)",
                    exc_info=e
                )
            return False, None
        except sql.OperationalError as e:
            logging.error(
                    "Operational Error at %s",
                    "ts_readdetections.py(check_query_status)",
                    exc_info=e
                )
            return False, None  
        except Exception as e:
            logging.error(
                    "Error at %s", "ts_readdetections.py(check_query_status)", exc_info=e
                )
            return False, None
        except RuntimeError as e:
            logging.error("Error at %s", "ts_readdetections.py(check_query_status)", exc_info=e)
            return False, None
        except SyntaxError as e:
            logging.error("Error at %s", "ts_readdetections.py(check_query_status)", exc_info=e)
            return False, None
       
# Function to get job run status using runs/list
def get_clusterstatusfromjob(job_id):
    if 'WEBSITE_SITE_NAME' in os.environ:
        data = json.loads(os.getenv('dbinstance'))
    else:     
        dbinstancefile = config_dir + "/config.dbinstance.json"
        with open(dbinstancefile, "r") as file:
            data = json.load(file)
    

    Config.DATABRICKS_INSTANCE = data["DATABRICKS_INSTANCE"]
    Config.WAREHOUSE_ID = data["WAREHOUSE_ID"]
    Config.SCHEMA = data["SCHEMA"]
    Config.CATALOG = data["CATALOG"]
    Config.TABLE = data["TABLE"]
    print("Table" + Config.TABLE)
    Config.SQL_STATEMENTS_ENDPOINT = f'https://{Config.DATABRICKS_INSTANCE}/api/2.0/sql/statements'
    # Set up the headers for authentication
    headers = {
            "Authorization": f"Bearer {Config.PERSONAL_ACCESS_TOKEN}"
        }

    # Define the parameters for the API call
     # Define the parameters for the API call, including active_only=false
    params = {
        'job_id': job_id,
        'expand_tasks': 'true',
        'active_only': 'true'
    }
   
    url = f'https://{Config.DATABRICKS_INSTANCE}/api/2.2/jobs/runs/list'
    # Make the request to the Databricks API
    response = requests.get(url, headers=headers, params=params)
    try:
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Databricks Cluster Status check API call failed. URL: {url}, Response: {response.text}"
        ) from e
     # ---------- Parse JSON ----------
    try:
        job_run_info = response.json()
    except ValueError as e:
        raise RuntimeError("Invalid JSON response from towerscout.py get_clusterstatusfromjob") from e
    # ---------- Validate response ----------
    runs = job_run_info.get("runs", [])
    if not runs:
        # No active runs → valid state
        return "TERMINATED"

    try:
        task = runs[0]["tasks"][0]

        if "existing_cluster_id" in task:
            cluster_id = task["existing_cluster_id"]
        elif "cluster_instance" in task:
            cluster_id = task["cluster_instance"]["cluster_id"]
        else:
            # raise KeyError("Cluster ID not found in task")
            return "TERMINATED"

    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Unexpected Databricks response structure from towerscout.py get_clusterstatusfromjob: {job_run_info}"
        ) from e

    # ---------- Business logic ----------
    if not cluster_id or cluster_id == "undefined":
        return "TERMINATED"

    return get_cluster_status_from_databricks(cluster_id)
# The job is running
def get_jobstatus(run_id):
    # trigger_time_ms = int(time.time() * 1000)
    # run_id = discover_run_id(job_id, request_id,trigger_time_ms)
    job_failed, jobrun_details = is_databricks_job_failed(run_id)
    return job_failed, jobrun_details


# Function to get the cluster status
def get_cluster_status_from_databricks(cluster_id):
    # Databricks API authentication headers
    headers = {
            "Authorization": f"Bearer {Config.PERSONAL_ACCESS_TOKEN}"
        }
    
    # Define the parameters to be passed to the API
    params = {'cluster_id': cluster_id}

    url = f'https://{Config.DATABRICKS_INSTANCE}/api/2.1/clusters/get'
    try:
        # Make the request to the Databricks API
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raises for 4xx / 5xx

        # # Check the response status code
        # if response.status_code == 200:
            # Successfully fetched cluster details
        cluster_info = response.json()  # May raise ValueError()
        if "state" not in cluster_info:
            raise KeyError("Missing 'state' in Databricks response")

        return cluster_info["state"]
        
    except requests.exceptions.RequestException as e:
        # Network errors, timeouts, HTTP errors
        raise RuntimeError(
            f"Failed to retrieve cluster status. Databricks cluster status request failed. "
            f"Cluster ID: {cluster_id}, URL: {url}, {response.json()}"
        ) from e

    except ValueError as e:
        # JSON decoding error
        raise RuntimeError(
            f"Failed to retrieve cluster status. Invalid JSON response from Databricks. Cluster ID: {cluster_id}, {response.json()}"
        ) from e

    except KeyError as e:
        # Unexpected response shape
        raise RuntimeError(
            f"Failed to retrieve cluster status. Unexpected Databricks response format: {e}, {response.json()}"
        ) from e
def get_today_utc_range_ms():
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow_start = today_start + datetime.timedelta(days=1)

    return (
        int(today_start.timestamp() * 1000),
        int(tomorrow_start.timestamp() * 1000)
    )

def discover_run_idOld(job_id, request_id, trigger_time_ms, max_retries=5, delay=5):
    """
    Discover the Databricks run_id corresponding to a user request.

    Args:
        job_id (int): Databricks job ID
        request_id (str): Unique ID for this user request
        trigger_time_ms (int): Epoch time in ms when the request was triggered
        max_retries (int): Number of retries if run_id not found
        delay (int): Delay between retries in seconds

    Returns:
        run_id (int) if found, None otherwise
    """
    url = f"https://{Config.DATABRICKS_INSTANCE}/api/2.2/jobs/runs/list"
    headers = {"Authorization": f"Bearer {Config.PERSONAL_ACCESS_TOKEN}"}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                params={"job_id": job_id, "active_only": "false", "limit": 10},
                timeout=10
            )

            if response.status_code != 200:
                logging.warning(f"Attempt {attempt}: Failed to list runs, status code {response.status_code}")
                time.sleep(delay)
                continue

            runs = response.json().get("runs", [])
            if not runs:
                logging.info(f"Attempt {attempt}: No runs found yet for job_id {job_id}")
                time.sleep(delay)
                continue

            for run in runs:
                # # Attempt to match by request_id first - this is not working
                # notebook_params = run.get("notebook_params", {})
                # if notebook_params.get("request_id") == request_id:
                #     return run["run_id"]

                # Fallback: match by start time
                start_time = run.get("start_time", 0)
                end_time_ms = run.get("end_time")
                start_time_ms = run.get("start_time")
                start_time = datetime.datetime.fromtimestamp(start_time_ms / 1000)
                is_today = start_time.date() == datetime.date.today()
                state = run.get("state", {})
                life_cycle = state.get("life_cycle_state")
                result_state = state.get("result_state")
                if life_cycle != "TERMINATED" and not end_time_ms and is_today:
                    return run["run_id"], False
                if life_cycle == "TERMINATED" and result_state == "FAILED" and is_today and (start_time>=trigger_time_ms):
                    return run["run_id"], True
            # Not found in this attempt
            logging.info(f"Attempt {attempt}: No matching run_id found yet.")
            time.sleep(delay)

        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt}: RequestException while listing runs: {e}")
            time.sleep(delay)
        except ValueError as e:
            logging.error(f"Attempt {attempt}: JSON decode error: {e}")
            time.sleep(delay)
        except KeyError as e:
            logging.error(f"Attempt {attempt}: Missing expected field: {e}")
            time.sleep(delay)
        except Exception as e:
            logging.error(f"Attempt {attempt}: Unexpected error: {e}")
            time.sleep(delay)

    logging.error(f"Failed to discover run_id for request_id {request_id} after {max_retries} attempts")
    return None
def discover_run_idwithstarttime(job_id, trigger_time_ms, max_retries=5, delay=5):
    """
    Returns (run_id, is_failed) or None
    Ensures the run started today (UTC).
    """

    url = f"https://{Config.DATABRICKS_INSTANCE}/api/2.2/jobs/runs/list"
    headers = {"Authorization": f"Bearer {Config.PERSONAL_ACCESS_TOKEN}"}

    today_start_ms, tomorrow_start_ms = get_today_utc_range_ms()

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                params={
                    "job_id": job_id,
                    "active_only": "false",
                    "limit": 20
                },
                timeout=10
            )

            if response.status_code != 200:
                logging.warning(
                    f"Attempt {attempt}: Failed to list runs "
                    f"(status {response.status_code})"
                )
                time.sleep(delay)
                continue

            runs = response.json().get("runs", [])

            for run in runs:
                start_time_ms = run.get("start_time")
                if not start_time_ms:
                    continue

                # ✅ Ensure run started today
                if not (today_start_ms <= start_time_ms < tomorrow_start_ms):
                    continue

                # Optional: also ensure it started after the trigger
                if start_time_ms < trigger_time_ms:
                    continue

                state = run.get("state", {})
                life_cycle = state.get("life_cycle_state")
                result_state = state.get("result_state")

                # 🟡 Still running
                if life_cycle in ("PENDING", "RUNNING", "TERMINATING"):
                    return run["run_id"], False

                # 🔴 Finished & failed
                if life_cycle == "TERMINATED" and result_state == "FAILED":
                    return run["run_id"], True

            logging.info(f"Attempt {attempt}: No matching run found yet")
            time.sleep(delay)

        except Exception as e:
            logging.error(f"Attempt {attempt}: Error while discovering run: {e}")
            time.sleep(delay)

    logging.error(f"No run started today found for job {job_id}")
    return None
def discover_run_id(job_id, trigger_time_ms, max_retries=5, delay=5):
    """
    Returns (run_id, is_failed) or None
    Ensures the run started today (UTC).
    """

    url = f"https://{Config.DATABRICKS_INSTANCE}/api/2.2/jobs/runs/list"
    headers = {"Authorization": f"Bearer {Config.PERSONAL_ACCESS_TOKEN}"}

    today_start_ms, tomorrow_start_ms = get_today_utc_range_ms()

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                params={
                    "job_id": job_id,
                    "active_only": "false",
                    "limit": 20
                },
                timeout=10
            )

            if response.status_code != 200:
                logging.warning(
                    f"Attempt {attempt}: Failed to list runs "
                    f"(status {response.status_code})"
                )
                time.sleep(delay)
                continue

            runs = response.json().get("runs", [])

            for run in runs:
                start_time_ms = run.get("start_time")
                if not start_time_ms:
                    continue

                # ✅ Ensure run started today
                if not (today_start_ms <= start_time_ms < tomorrow_start_ms):
                    continue

                # Optional: also ensure it started after the trigger
                if start_time_ms < trigger_time_ms:
                    continue

                state = run.get("state", {})
                life_cycle = state.get("life_cycle_state")
                result_state = state.get("result_state")

                # 🟡 Still running
                if life_cycle in ("PENDING", "RUNNING", "TERMINATING"):
                    return run["run_id"], False

                # 🔴 Finished & failed
                if life_cycle == "TERMINATED" and result_state == "FAILED":
                    return run["run_id"], True

            logging.info(f"Attempt {attempt}: No matching run found yet")
            time.sleep(delay)

        except Exception as e:
            logging.error(f"Attempt {attempt}: Error while discovering run: {e}")
            time.sleep(delay)

    logging.error(f"No run started today found for job {job_id}")
    return None


def is_databricks_job_failed(run_id):
    # url = f"{Config.DATABRICKS_INSTANCE}/api/2.1/jobs/runs/get"
    # headers = {
    #     "Authorization": f"Bearer {Config.PERSONAL_ACCESS_TOKEN}"
    # }
    # params = {
    #     "run_id": run_id
    # }

    # response = requests.get(url, headers=headers, params=params)
    try:
        response = requests.post(
        f"https://{Config.DATABRICKS_INSTANCE}/api/2.1/jobs/runs/get",
        headers={
            "Authorization": f"Bearer {Config.PERSONAL_ACCESS_TOKEN}",
            "Content-Type": "application/json"
        },
        json={"run_id": int(run_id)},
        timeout=10
    )
        response.raise_for_status()
        run_info = response.json()

        state = run_info.get("state", {})
        life_cycle = state.get("life_cycle_state")
        result_state = state.get("result_state")
    

        if life_cycle == "TERMINATED" and result_state == "FAILED":
            run_details = {
            "life_cycle_state": state.get("life_cycle_state"),
            "result_state": state.get("result_state"),
            "state_message": state.get("state_message"),
            "start_time": run_info.get("start_time"),
            "end_time": run_info.get("end_time"),
            "setup_duration_ms": run_info.get("setup_duration"),
            "execution_duration_ms": run_info.get("execution_duration"),
            "run_duration_ms": run_info.get("run_duration")
            }
            return True, run_details

        if life_cycle == "TERMINATED" and result_state == "SUCCESS":
            return False, None

        return None, None  # still running
    except Exception as e:
        logging.error(
            "Error at %s", "is_databricks_job_failed", exc_info=e
        )
        raise
    except RuntimeError as e:
        logging.error("Error at %s", "is_databricks_job_failed", exc_info=e)
        raise
    except SyntaxError as e:
        logging.error("Error at %s", "is_databricks_job_failed", exc_info=e)
        raise
    except requests.exceptions.RequestException as e:
        logging.error("Databricks Job status API call failed", exc_info=e)
        raise
def get_job_run_details(run_id):
    response = requests.get(
        f"https://{Config.DATABRICKS_INSTANCE}/api/2.1/jobs/runs/get",
        headers={"Authorization": f"Bearer {Config.PERSONAL_ACCESS_TOKEN}"},
        params={"run_id": run_id},
        timeout=10
    )
    response.raise_for_status()

    run = response.json()
    state = run.get("state", {})

    details = {
        "life_cycle_state": state.get("life_cycle_state"),
        "result_state": state.get("result_state"),
        "state_message": state.get("state_message"),
        "start_time": run.get("start_time"),
        "end_time": run.get("end_time"),
        "setup_duration_ms": run.get("setup_duration"),
        "execution_duration_ms": run.get("execution_duration"),
        "run_duration_ms": run.get("run_duration")
    }

    return details

def get_job_run_error_output(run_id):
    response = requests.get(
        f"https://{Config.DATABRICKS_INSTANCE}/api/2.1/jobs/runs/get-output",
        headers={"Authorization": f"Bearer {Config.PERSONAL_ACCESS_TOKEN}"},
        params={"run_id": run_id},
        timeout=10
    )

    if response.status_code != 200:
        return None

    output = response.json()

    # Notebook task
    if "error" in output:
        return output["error"]

    # Spark / JAR / Python task
    if "metadata" in output and "error" in output["metadata"]:
        return output["metadata"]["error"]

    return None
