from databricks import sql
from requests.exceptions import Timeout, RequestException
import os, time, logging, requests, ts_secrets, json, asyncio
from databricks.sql.exc import RequestError
from numpy.random import choice
config_dir = os.path.join(os.getcwd().replace("webapp", ""), "webapp")

DATABRICKS_INSTANCE = ""
PERSONAL_ACCESS_TOKEN = ts_secrets.getSecret('DB-PERSONAL-ACCESS-TOKEN')
WAREHOUSE_ID = ""
SQL_STATEMENTS_ENDPOINT = ""
CATALOG = ""
SCHEMA = ""
TABLE = ""
GOLD_TABLE = ""
class SilverTable:
    
                    # # # Testing existing data
                    # user_id = 'cnu4'
                    # request_id = '008d35a3'
                    
    def __init__(self):
        self.batch_size = 100
        if 'WEBSITE_SITE_NAME' in os.environ:
            data = json.loads(os.getenv('dbinstance'))
        else:     
            dbinstancefile = config_dir + "/config.dbinstance.json"
            with open(dbinstancefile, "r") as file:
                data = json.load(file)
        global DATABRICKS_INSTANCE
        global WAREHOUSE_ID
        global SQL_STATEMENTS_ENDPOINT
        global SCHEMA
        global CATALOG
        global TABLE

        DATABRICKS_INSTANCE = data["DATABRICKS_INSTANCE"]
        WAREHOUSE_ID = data["WAREHOUSE_ID"]
        SCHEMA = data["SCHEMA"]
        CATALOG = data["CATALOG"]
        TABLE = data["TABLE"]
        SQL_STATEMENTS_ENDPOINT = f'https://{DATABRICKS_INSTANCE}/api/2.0/sql/statements'

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
                    # # Testing existing data
                    # user_id = 'cnu4'
                    # request_id = '008d35a3'
                    sql_query  = "SELECT bboxes from edav_prd_csels.towerscout.test_image_silver WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'"
                    # Set the payload for the request (include the query)
                    data = {
                        'statement': sql_query,
                        'warehouse_id': WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
                    }
                    # Set the headers with the personal access token for authentication
                    headers = {
                        'Authorization': f'Bearer {PERSONAL_ACCESS_TOKEN}',
                        'Content-Type': 'application/json',
                    }
            

                    response = requests.post(SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
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
            
    # @retry(wait=wait_exponential(min=2, max=10), stop=stop_after_attempt(3))
    def fetchbboxesOld(self, request_id, user_id, tile_count):
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
                        "SELECT bboxes from " + CATALOG + "." + SCHEMA + "."  + TABLE + " WHERE user_id = '"
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
                    # # Testing existing data
                    # user_id = 'cnu4'
                    # request_id = '008d35a3'
                    sql_query  = (f"SELECT bboxes, "
                    "concat(uuid,'.jpeg') AS filename, "
                    "image_path AS url, "
                    "uuid, image_hash "
                    "FROM " + CATALOG + "." + SCHEMA + "." + TABLE + " where user_id = '" + user_id + "' AND request_id = '" + request_id + "' "
                    "ORDER BY image_id")
                    print(f"sql_query: {sql_query}")
                    
                     # Set the payload for the request (include the query)
                    data = {
                        'statement': sql_query,
                        'warehouse_id': WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
                        'output_format': 'json'
                    }
                    # Set the headers with the personal access token for authentication
                    headers = {
                        'Authorization': f'Bearer {PERSONAL_ACCESS_TOKEN}',
                        'Content-Type': 'application/json',
                    }
            
                    response = requests.post(SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
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
       
    def fetchbboxesWithoutPollingold(self, request_id, user_id, tiles_count):
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
                        "SELECT bboxes, concat(uuid,'.jpeg') AS filename, image_path AS url from " + CATALOG + "." + SCHEMA + "."  + TABLE + " WHERE user_id = '"
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
                    #     FROM " + CATALOG + "." + SCHEMA + "."  + TABLE + " where request_id = ? AND user_id = ?
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
            
            sql_query  = (f"SELECT count(bboxes) "
                          "FROM " + CATALOG + "." + SCHEMA + "." + TABLE + "  WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'")
# Set the payload for the request (include the query)
            data = {
            'statement': sql_query,
            'warehouse_id': WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
}
        # Set the headers with the personal access token for authentication
            headers = {
            'Authorization': f'Bearer {PERSONAL_ACCESS_TOKEN}',
            'Content-Type': 'application/json',
        }
            while retries < max_retries:
                try:
                    # if connection is None:
                    #      time.sleep(60)
                        #  continue
                    # if not connection.open:
                    response = requests.post(SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
                    print(f"SELECT count(bboxes) from " + CATALOG + "." + SCHEMA + "." + TABLE + " WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'")
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
        self, request_id, user_id, tile_count, max_retries, delay=20
        ):
        print("calling poll_SilverTableJobDoneWithLogs")
        # # Testing existing data
        # user_id = 'cnu4'
        # request_id = '7fe1cd27'
        try:
        
            retries = 0
            global totalTilesProcessed
            totalTilesProcessed = 0
            sql_query  = "SELECT count(bboxes) from " + CATALOG + "." + SCHEMA + "."  + TABLE + " WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'"
# Set the payload for the request (include the query)
            data = {
            'statement': sql_query,
            'warehouse_id': WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
        }
        # Set the headers with the personal access token for authentication
            headers = {
            'Authorization': f'Bearer {PERSONAL_ACCESS_TOKEN}',
            'Content-Type': 'application/json',
        }
            while retries < max_retries:
                try:
                    # if connection is None:
                    #      time.sleep(60)
                        #  continue
                    # if not connection.open:
                    response = requests.post(SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
                    print(f"SELECT count(bboxes) from " + CATALOG + "." + SCHEMA + "."  + TABLE + " WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'")
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
                    logging.info(f"result count:{newTilesProcessCount}")
                    logging.error("SyntaxError at %s","ts_readdetections.py(poll_SilverTableJobDone)",
                    exc_info=e)
                    time.sleep(delay)
                except RequestException as e:
                    print("RequestException silvertablejobdone. Retrying...")
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
    
    def get_bboxesfortilesWithoutPollingOld(self, tiles, events, id, request_id, user_id):
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
                "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}",
                "Content-Type": "application/json"
            }
    
            body = {
                "warehouse_id": WAREHOUSE_ID,
                "statement": sql_query,
                "catalog": catalog,
                "schema": schema
            }
    
            response = requests.post(SQL_STATEMENTS_ENDPOINT, json=body, headers=headers)
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
        url = f'https://{DATABRICKS_INSTANCE}/api/2.0/sql/statements/{statement_id}'
        headers = {
            "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}"
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
    
                    # # # Testing existing data
                    # user_id = 'cnu4'
                    # request_id = '008d35a3'
                    
    def __init__(self):
        self.batch_size = 100
        if 'WEBSITE_SITE_NAME' in os.environ:
            data = json.loads(os.getenv('dbinstance'))
        else:     
            dbinstancefile = config_dir + "/config.dbinstance.json"
            with open(dbinstancefile, "r") as file:
                data = json.load(file)
        global DATABRICKS_INSTANCE
        global WAREHOUSE_ID
        global SQL_STATEMENTS_ENDPOINT
        global SCHEMA
        global CATALOG
        global TABLE
        global GOLD_TABLE


        DATABRICKS_INSTANCE = data["DATABRICKS_INSTANCE"]
        WAREHOUSE_ID = data["WAREHOUSE_ID"]
        SCHEMA = data["SCHEMA"]
        CATALOG = data["CATALOG"]
        TABLE = data["TABLE"]
        GOLD_TABLE = data["GOLD_TABLE"]
        SQL_STATEMENTS_ENDPOINT = f'https://{DATABRICKS_INSTANCE}/api/2.0/sql/statements'

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
                    # # Testing existing data
                    # user_id = 'cnu4'
                    # request_id = '008d35a3'
                    sql_query  = "SELECT bboxes from edav_prd_csels.towerscout.test_image_gold WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'"
                    # Set the payload for the request (include the query)
                    data = {
                        'statement': sql_query,
                        'warehouse_id': WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
                    }
                    # Set the headers with the personal access token for authentication
                    headers = {
                        'Authorization': f'Bearer {PERSONAL_ACCESS_TOKEN}',
                        'Content-Type': 'application/json',
                    }
            

                    response = requests.post(SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
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
            
    # @retry(wait=wait_exponential(min=2, max=10), stop=stop_after_attempt(3))
    def fetchbboxesOld(self, request_id, user_id, tile_count):
        logging.info("Startinng ts_readdetections.py(fetchbboxes)")
        attempt = 0
        retries = 3
        connection = None
        cursor = None
        while attempt < retries:
            try:
                max_retries = tile_count * 2
                jobdone = self.poll_GoldTableJobDone(
                    request_id, user_id, tile_count, max_retries
                )
            
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
                        "SELECT bboxes from " + CATALOG + "." + SCHEMA + "."  + TABLE + " WHERE user_id = '"
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
                    # # Testing existing data
                    # user_id = 'cnu4'
                    # request_id = '008d35a3'
                    sql_query  = (f"SELECT bboxes, "
                    "concat(uuid,'.jpeg') AS filename, "
                    "image_path AS url, "
                    "uuid, image_hash "
                    "FROM " + CATALOG + "." + SCHEMA + "." + GOLD_TABLE + " where user_id = '" + user_id + "' AND request_id = '" + request_id + "' "
                    "ORDER BY image_id")
                    print(f"sql_query: {sql_query}")
                    
                     # Set the payload for the request (include the query)
                    data = {
                        'statement': sql_query,
                        'warehouse_id': WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
                        'output_format': 'json'
                    }
                    # Set the headers with the personal access token for authentication
                    headers = {
                        'Authorization': f'Bearer {PERSONAL_ACCESS_TOKEN}',
                        'Content-Type': 'application/json',
                    }
            
                    response = requests.post(SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
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
       
    def fetchbboxesWithoutPollingold(self, request_id, user_id, tiles_count):
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
                        "SELECT bboxes, concat(uuid,'.jpeg') AS filename, image_path AS url from " + CATALOG + "." + SCHEMA + "."  + TABLE + " WHERE user_id = '"
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
                    #     FROM " + CATALOG + "." + SCHEMA + "."  + TABLE + " where request_id = ? AND user_id = ?
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
    def poll_GoldTableJobDone(
        self, request_id, user_id, tile_count, max_retries, delay=10
    ):
        
        # # Testing existing data
        # user_id = 'cnu4'
        # request_id = '7fe1cd27'
        try:
        
            retries = 0
            
            sql_query  = (f"SELECT count(bboxes) "
                          "FROM " + CATALOG + "." + SCHEMA + "." + TABLE + "  WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'")
# Set the payload for the request (include the query)
            data = {
            'statement': sql_query,
            'warehouse_id': WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
}
        # Set the headers with the personal access token for authentication
            headers = {
            'Authorization': f'Bearer {PERSONAL_ACCESS_TOKEN}',
            'Content-Type': 'application/json',
        }
            while retries < max_retries:
                try:
                    # if connection is None:
                    #      time.sleep(60)
                        #  continue
                    # if not connection.open:
                    response = requests.post(SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
                    print(f"SELECT count(bboxes) from " + CATALOG + "." + SCHEMA + "." + TABLE + " WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'")
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
        # # Testing existing data
        # user_id = 'cnu4'
        # request_id = '7fe1cd27'
        try:
        
            retries = 0
            global totalTilesProcessed
            totalTilesProcessed = 0
            sql_query  = "SELECT count(bboxes) from " + CATALOG + "." + SCHEMA + "."  + TABLE + " WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'"
# Set the payload for the request (include the query)
            data = {
            'statement': sql_query,
            'warehouse_id': WAREHOUSE_ID,  # Replace with your actual SQL warehouse ID
        }
        # Set the headers with the personal access token for authentication
            headers = {
            'Authorization': f'Bearer {PERSONAL_ACCESS_TOKEN}',
            'Content-Type': 'application/json',
        }
            while retries < max_retries:
                try:
                    # if connection is None:
                    #      time.sleep(60)
                        #  continue
                    # if not connection.open:
                    response = requests.post(SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)
                    print(f"SELECT count(bboxes) from " + CATALOG + "." + SCHEMA + "."  + TABLE + " WHERE user_id = '" + user_id + "' AND request_id = '" + request_id + "'")
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
    
    def get_bboxesfortilesWithoutPollingOld(self, tiles, events, id, request_id, user_id):
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
                "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}",
                "Content-Type": "application/json"
            }
    
            body = {
                "warehouse_id": WAREHOUSE_ID,
                "statement": sql_query,
                "catalog": catalog,
                "schema": schema
            }
    
            response = requests.post(SQL_STATEMENTS_ENDPOINT, json=body, headers=headers)
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
        url = f'https://{DATABRICKS_INSTANCE}/api/2.0/sql/statements/{statement_id}'
        headers = {
            "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}"
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
    global DATABRICKS_INSTANCE
    global WAREHOUSE_ID
    global SQL_STATEMENTS_ENDPOINT
    global SCHEMA
    global CATALOG
    global TABLE

    DATABRICKS_INSTANCE = data["DATABRICKS_INSTANCE"]
    WAREHOUSE_ID = data["WAREHOUSE_ID"]
    SCHEMA = data["SCHEMA"]
    CATALOG = data["CATALOG"]
    TABLE = data["TABLE"]
    print("Table" + TABLE)
    SQL_STATEMENTS_ENDPOINT = f'https://{DATABRICKS_INSTANCE}/api/2.0/sql/statements'
    # Set up the headers for authentication
    headers = {
            "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}"
        }

    # Define the parameters for the API call
     # Define the parameters for the API call, including active_only=false
    params = {
        'job_id': job_id,
        'expand_tasks': 'true',
        'active_only': 'true'
    }
   
    url = f'https://{DATABRICKS_INSTANCE}/api/2.2/jobs/runs/list'
    # Make the request to the Databricks API
    response = requests.get(url, headers=headers, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Successfully retrieved job run details
        job_run_info = response.json()
        if (len(job_run_info)>0):
            if(('existing_cluster' in job_run_info['runs'][0]['tasks'][0])):
                clusterID = job_run_info['runs'][0]['tasks'][0]['existing_cluster_id']
            elif(('cluster_instance' in job_run_info['runs'][0]['tasks'][0])):
                clusterID = job_run_info['runs'][0]['tasks'][0]['cluster_instance']['cluster_id']
            if(clusterID == 'undefined'):
                return "TERMINATED"
            else:
                return get_cluster_status_from_databricks(clusterID)
        else:
            return "TERMINATED"
            return {"error": f"Failed to retrieve job run status, url:{url}, Token: {PERSONAL_ACCESS_TOKEN}", "message": response.json()}
        print(len(job_run_info))
        print(job_run_info['runs'][0]['setup_duration'])
        print(job_run_info['runs'][0]['state'])
        print(job_run_info['runs'][0]['tasks'][0]['existing_cluster_id'])
        # active_runs = [run for run in job_run_info['runs'] if run['state']['life_cycle_state'] == 'RUNNING']
        # Non_active_runs = [run for run in job_run_info['runs']]
        
        # print((job_run_info))
        # print(len(active_runs))
        # print(len(Non_active_runs))
        # # print(job_run_info['runs'])
        # return job_run_info
    else:
        # If the request fails, return an error message
        return {"error": f"Failed to retrieve job run status, url:{url}, Token: {PERSONAL_ACCESS_TOKEN}", "message": response.json()}

# Function to get the cluster status
def get_cluster_status_from_databricks(cluster_id):
    # Databricks API authentication headers
    headers = {
            "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}"
        }
    
    # Define the parameters to be passed to the API
    params = {'cluster_id': cluster_id}

    url = f'https://{DATABRICKS_INSTANCE}/api/2.1/clusters/get'
    # Make the request to the Databricks API
    response = requests.get(url, headers=headers, params=params)

    # Check the response status code
    if response.status_code == 200:
        # Successfully fetched cluster details
        cluster_info = response.json()
        print(cluster_info["state"])
        return cluster_info["state"]
    else:
        # Return error message if the API request fails
        return {"error": "Failed to retrieve cluster status", "message": response.json()}
