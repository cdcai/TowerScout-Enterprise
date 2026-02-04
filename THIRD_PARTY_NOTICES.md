# Third-Party Services and Open-Source Libraries

## Proprietary Azure Cloud Services

This application uses Microsoft Azure cloud services, including:

- Azure Maps APIs  
- Azure Key Vault  
- Azure Storage  
- Azure Databricks  

These services are **proprietary** and are not included with this project.  
Users must obtain a valid Microsoft Azure subscription and comply with Microsoft’s terms of service.  

No Azure subscription keys or credentials are included.

---

## Open-Source Python Libraries

This project uses open-source Python libraries installed via `requirements.txt`. Examples include:

| Library | License | Repository / URL |
|---------|---------|----------------|
| azure-core | Apache 2.0 | https://github.com/Azure/azure-sdk-for-python |
| msrest | Apache 2.0 | https://github.com/Azure/msrest-for-python |
| requests | Apache 2.0 | https://github.com/psf/requests |
| azure-identity | MIT | https://github.com/Azure/azure-sdk-for-python |
| numpy | BSD | https://numpy.org/ |
| pandas | BSD | https://pandas.pydata.org/ |

> For the full list of dependencies, see `requirements.txt`.

---

## Machine Learning Libraries

| Library | License | Repository / URL |
|---------|---------|----------------|
| PyTorch | BSD 3-Clause | https://pytorch.org/ |
| YOLO | MIT | https://docs.ultralytics.com/models/yolov5/ |

These libraries are used under their respective open-source licenses.

---

## Notes

- This file **does not include Azure subscription keys or credentials**.  
- Apache 2.0 dependencies included via `pip install` do **not require a separate NOTICE file** unless you redistribute the libraries.  
- Users are responsible for complying with the licenses of all third-party libraries and services.
