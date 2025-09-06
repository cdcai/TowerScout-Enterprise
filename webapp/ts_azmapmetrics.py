from azure.identity import ClientSecretCredential
import calendar
from datetime import datetime
import requests
import ts_secrets
from requests.exceptions import ConnectionError, Timeout
from azure.core.exceptions import ClientAuthenticationError
from azure.mgmt.resource import SubscriptionClient

class azTransactions:
   
 api_version = '2024-02-01'  # Use the appropriate API version
   
   
 def getAZTransactionCount(intEnv):
  try:  
   
     # Azure Maps account details
   if intEnv == 1:
        # Production
     subscription_id = ts_secrets.getSecret('TSAZMAPSUBSCRID')
     resource_group_name = ts_secrets.getSecret('TSAZMAPKEYRG')
     account_name = ts_secrets.getSecret('TSAZMAPACCOUNTNAME')
     tenant_id = ts_secrets.getSecret('TSAZMAPACCNTSPTENANTID')
     client_id = ts_secrets.getSecret('TSAZMAPACCNTSPCLIENTID')
     client_secret = ts_secrets.getSecret('TSAZMAPACCNTSPCLIENTSECRET')
   elif intEnv == 2:
    # Dev
     subscription_id = ts_secrets.getSecret('TSAZMAPSUBSCRID')
     resource_group_name = ts_secrets.getSecret('TSAZMAPKEYRG')
     account_name = ts_secrets.getSecret('TSAZMAPACCOUNTNAME')
     tenant_id = ts_secrets.getSecret('TSAZMAPACCNTSPTENANTID')
     client_id = ts_secrets.getSecret('TSAZMAPACCNTSPCLIENTID')
     client_secret = ts_secrets.getSecret('TSAZMAPACCNTSPCLIENTSECRET')
   
   # Define the time range and metrics to filter
   now = datetime.today()
   lastDayofMonth = calendar.monthrange(now.year,now.month)[1]
   # month_end_date=datetime(today.year,today.month,1) + datetime.timedelta(days=calendar.monthrange(today.year,today.month)[1] - 1)
   timespan=str(now.year) + '-' + str(now.month) + '-01T00:00:00z/' + str(now.year) + '-' + str(now.month) + '-' + str(lastDayofMonth) + 'T23:59:59z' 

   # Azure Metrics API endpoint
   metrics_api_url = f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.Maps/accounts/{account_name}/providers/microsoft.insights/metrics"

   interval = 'P1D'  # 1 day interval

   # Get the access token
   credential = ClientSecretCredential(tenant_id, client_id, client_secret)
   token = credential.get_token("https://management.azure.com/.default")

   # Set the query parameters
   params = {
     'metricnames': 'Usage',  # Specify the metrics you want
     'timespan': timespan,  # Define the time range for metrics
     # Examples: PT15M, PT1H, P1D, FULL
     'interval': 'FULL',  # Optional: e.g., 1-day intervals
     # 'aggregation': 'Total,Average',  # Specify aggregation methods
     'api-version': azTransactions.api_version   # API version
    }
     
   # Set authorization headers
   headers = {
    'Authorization': f'Bearer {token.token}',
    'Content-Type': 'application/json'
}

   # Make the GET request
   response = requests.get(metrics_api_url, headers=headers, params=params,timeout=600)

   # Check the response
   if response.status_code == 200:
        metrics_data = response.json()
   else:
      print(f"Error: {response.status_code}, {response.text}")
      raise RuntimeError("Error '" + response.status_code + ": " + response.text + "' occured at in ts_azmapMetircs.py while getting Azure maps metrics")
   
 
   total_count = 0
  
   if "value" in metrics_data and len(metrics_data["value"]) > 0:
    for metric in metrics_data["value"]:
        for time_series in metric["timeseries"]:
            for data_point in time_series["data"]:
                    # Accumulate the total counts
                    daily_total = data_point.get("count") or data_point.get("average") or 0
                    total_count += daily_total
    # Get the month name
    
    month_name = now.strftime("%B")
    current_year = now.year
    return (month_name + " " + str(current_year) + " Azure map transaction count: " + str(total_count))
   else:
      return("")
    
  except ClientAuthenticationError as e:
    raise ClientAuthenticationError("Error '" + e + "' occured at in ts_azmapMetircs.py while getting Azure maps metrics")
  except RuntimeError as e:
    raise RuntimeError("Error '" + e + "' occured at in ts_azmapMetircs.py while getting Azure maps metrics")
  except Exception as e:
    raise RuntimeError("Error '" + e + "' occured at in ts_azmapMetircs.py while getting Azure maps metrics")
  except ConnectionError as e:
    raise ConnectionError("Connection Error '" + e + "' occured at in ts_azmapMetircs.py while getting Azure maps metrics")
  except Timeout as e:
    raise Timeout("Timeout Error '" + e + "' occured at in ts_azmapMetircs.py while getting Azure maps metrics")
  except requests.exceptions.RequestException as e:
    raise requests.exceptions.RequestException("RequestException Error '" + e + "' occured at in ts_azmapMetircs.py while getting Azure maps metrics")
    
 