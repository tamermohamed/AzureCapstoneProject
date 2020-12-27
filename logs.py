from azureml.core import Workspace
from azureml.core.webservice import Webservice


class LogService:
   
   def Log(serviceName):

        # Requires the config to be downloaded first to the current working directory
        ws = Workspace.from_config()

        # load existing web service
        service = Webservice(name=serviceName, workspace=ws)

        service.update(enable_app_insights=True)

        logs = service.get_logs()

        for line in logs.split('\n'):
            print(line)
