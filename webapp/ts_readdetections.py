from databricks import sql
from requests.exceptions import Timeout, RequestException
import os, time, logging


class SilverTable:
    def __init__(self):
        self.batch_size = 100

    def fetchbboxes(self, request_id, user_id, tile_count):
        print("request_id", request_id)
        print("request_id", user_id)
        print("tilecount", tile_count)
        max_retries = tile_count * 2
        jobdone = self.poll_SilverTableJobDone(
                request_id, user_id, tile_count, max_retries
            )
        # jobdone = poll_table_for_record_count(
        #    request_id, user_id, tile_count, max_retries
        #    )
        if jobdone:
            connection = sql.connect(
                server_hostname="adb-1881246389460182.2.azuredatabricks.net",
                http_path="/sql/1.0/warehouses/8605a48953a7f210",
                access_token="dapicb010df06931117a00ccc96cab0abdf0-3",
            )
            # # Testing existing data
            # user_id = 'cnu4'
            # request_id = '32c219ef' # azure 10 mtrs
            try:
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
            except sql.DatabaseError as e:
                logging.error(
                    "Database Error at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                )
            except sql.OperationalError as e:
                logging.error(
                    "Operational Error at %s",
                    "ts_readdetections.py(fetchbboxes)",
                    exc_info=e
                )
            except Exception as e:
                cursor.close()
                connection.close()
                logging.error(
                    "Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e
                )
            except RuntimeError as e:
                logging.error("Error at %s", "ts_readdetections.py(fetchbboxes)", exc_info=e)

    def poll_SilverTableJobDone(
        self, request_id, user_id, tile_count, max_retries, delay=30
    ):
        
        # # Testing existing data
        # user_id = 'cnu4'
        # request_id = '7fe1cd27'
        try:
            connection = sql.connect(
            server_hostname="adb-1881246389460182.2.azuredatabricks.net",
            http_path="/sql/1.0/warehouses/8605a48953a7f210",
            access_token="dapicb010df06931117a00ccc96cab0abdf0-3",
            connection_timeoutv= 30,  # Timeout in seconds
            retry_config={
            "min_retry_delay": 1.0,
            "max_retry_delay": 60.0,
            "max_attempts": max_retries,
            "retry_duration": 900.0,
            "default_retry_delay": 5.0,
            }
        )
            cursor = connection.cursor()
            retries = 0
            print("retries", max_retries)
            while retries < max_retries:
                try:
                    if connection is None:
                         time.sleep(delay*2)
                         continue
                    cursor.execute(
                        "SELECT count(bboxes) from edav_dev_csels.towerscout.test_image_silver WHERE user_id = '"
                        + user_id
                        + "' AND request_id = '"
                        + request_id
                        + "'"
                    )
                    print(
                        f"SELECT count(bboxes) from edav_dev_csels.towerscout.test_image_silver WHERE user_id = '"
                        + user_id
                        + "' AND request_id = '"
                        + request_id
                        + "'"
                    )
                    resultcount = cursor.fetchone()
                    print("result count:", resultcount)
                    logging.info(f"result count:{resultcount}")
                    if resultcount[0] >= tile_count:
                        cursor.close()
                        connection.close()
                        print("result count:", resultcount)
                        return True
                    else:
                        retries += 1
                        if retries == max_retries:
                            max_retries +=1
                            # raise Timeout("Forcing timeout error")
                        time.sleep(delay)
                except Timeout as e:
                    print("Request timeout error occurred. Retrying...")
                    retries += 1
                    if retries >= max_retries:
                        max_retries+=1
                    time.sleep(delay)  # Wait before retrying
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
            cursor.close()
            connection.close()
            logging.error(
                "Error at %s",
                "ts_readdetections.py(poll_SilverTableJobDone)",
                exc_info=e
            )
        except RuntimeError as e:
            logging.error("Error at %s", "ts_readdetections.py(poll_SilverTableJobDone)", exc_info=e)
        

    def get_bboxesfortiles(self, tiles, events, id, request_id, user_id):
        tile_count = len(tiles)
        results_raw = self.fetchbboxes(request_id, user_id, tile_count)
        # print("result raw", results_raw)
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


# Function to query the table and get record count
def get_record_count(conn, user_id, request_id):
    cursor = conn.cursor()
    cursor.execute(
                        "SELECT count(bboxes) from edav_dev_csels.towerscout.test_image_silver WHERE user_id = '"
                        + user_id
                        + "' AND request_id = '"
                        + request_id
                        + "'"
                    )
    result = cursor.fetchone()
    return result[0]  # Return the record count from the query result

# Function to poll the table until the required record count is met
def poll_table_for_record_count(request_id, user_id, tile_count, max_retries, timeout = 300, delay=30):
    start_time = time.time()
    retries = 0

    # Connect to Databricks SQL Warehouse
    try:
        conn = sql.connect(
             server_hostname="adb-1881246389460182.2.azuredatabricks.net",
            http_path="/sql/1.0/warehouses/8605a48953a7f210",
            access_token="dapicb010df06931117a00ccc96cab0abdf0-3",
            timeout=30  # Set a connection timeout of 30 seconds
        )
    except RequestException as e:
        print(f"Connection error: {e}")
        return False

    while True:
        try:
            # Get the current record count from the table
            current_record_count = get_record_count(conn, user_id, request_id)

            # Check if the record count meets or exceeds the target
            if current_record_count >= tile_count:
                return True

            # If the timeout is exceeded, raise an error
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                return False

            # Wait before polling again
            time.sleep(delay)

        except Timeout:
            print("Request timeout error occurred. Retrying...")
            retries += 1
            if retries >= max_retries:
                print(f"Max retries reached. Aborting polling after {retries} attempts.")
                return None
            time.sleep(delay)  # Wait before retrying
        except RequestException as e:
            print(f"Request error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
        finally:
            # Always close the connection when done
            if conn:
                conn.close()

# # Example usage
# table_name = "your_table_name"  # Replace with your table name
# record_count = poll_table_for_record_count(request_id, user_id, tile_count, max_retries, delay=30)

# if record_count is not None:
#     print(f"Polling successful! Table has {record_count} records.")
# else:
#     print("Polling failed or timed out.")