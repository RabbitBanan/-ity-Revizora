from geopy.geocoders import Nominatim


geolocator = Nominatim(user_agent="coordinates_finder")
location = geolocator.geocode("Ozersk")
print(location.address)
print((location.latitude, location.longitude))
