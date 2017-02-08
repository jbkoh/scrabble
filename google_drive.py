from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
auth_url = gauth.GetAuthUrl() # Create authentication url user needs to visit
print "Visit here: %s"%auth_url
#gauth.Auth(code) # Authorize and build service from the code

code = raw_input()
gauth.Auth(code)

gdrive = GoogleDrive(gauth)
