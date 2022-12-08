'''
Utilitiy functions
'''
import geocoder
import json
from shapely.geometry import shape, Point
import pandas as pd

def geocode_dest(dest):
  """Find the latitude and longitude of input destination/address
  Parameters
  ----------
  dest: input destination/address
    
  Returns
  -------
  Latitude and longitude of input destination/address. 
  """
  g = geocoder.osm(dest)
  if g.ok:
    result = g.json
    return result['lat'], result['lng']
  else:
    return None, None

def geocode_geojson(lat, lng, boundaries):
  """Find the Chicago Community Area for a given input latitude and longitude
  Parameters
  ----------
  lat: input latitude
  lng: input longitude
  boundaries: loaded geospatial data with boundaries e.g., loaded from Boundaries - Community Areas (current).geojson
    
  Returns
  -------
  Community Area of input latitude and longitude. None if not found e.g., outside of Chicago.
  """
  if (lat is not None) and (lng is not None):
    for feature in boundaries['features']:
      polygon = shape(feature['geometry'])
      if polygon.contains(Point(lng, lat)):
        return feature['properties']['area_num_1']
  else:
    return None

def geocode_geojson_beat(lat, lng, boundaries):
  """Find the Chicago Police Beat for a given input latitude and longitude
  Parameters
  ----------
  lat: input latitude
  lng: input longitude
  boundaries: loaded geospatial data with boundaries e.g., loaded from Boundaries - Police Beats (current).geojson
    
  Returns
  -------
  Police Beat of input latitude and longitude. None if not found e.g., outside of Chicago.
  """
  if (lat is not None) and (lng is not None):
    for feature in boundaries['features']:
      polygon = shape(feature['geometry'])
      if polygon.contains(Point(lng, lat)):
        return int(feature['properties']['beat_num'])
  else:
    return None

def get_crime_types(clf, le, X_test_sample, pthr=0.1):
  """Return labels and predicted probabilities with probabalities > threshold
  Parameters
  ----------
  clf: Trained classifier with predict_proba method.
  
  le: Fitted (crime type) label encoder.

  X_test_sample: DataFrame of features with feature name. Only a single sample/row expected.

  pthr: probability threshold.
  
  Returns
  -------
  Dict of labels and predicted probabilities with probabalities > threshold. If multiple samples are present, only returns results from the first sample.
  """
  df = pd.DataFrame(clf.predict_proba(X_test_sample), columns=le.classes_)
  return df.loc[0][df.loc[0] >= pthr].sort_values(ascending=False).to_dict()
