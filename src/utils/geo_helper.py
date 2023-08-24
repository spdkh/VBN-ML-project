import math


# @title Mathematical Conversions
def spherical_mercator_to_lat_lon(x, y):
    # todo: needs to be fixed
    RADIUS_EARTH = 6378137.0  # Earth's radius in meters
    TILE_SIZE = 256.0  # Size of a tile in Web Mercator projection

    # Reverse the scaling and shifting applied during conversion to Web Mercator
    # x = x * TILE_SIZE - RADIUS_EARTH
    # y = y * TILE_SIZE - RADIUS_EARTH

    # Convert x and y to latitude and longitude in radians
    lon_rad = math.radians(x / RADIUS_EARTH)
    lat_rad = math.atan(math.exp(y / RADIUS_EARTH))
    lat_rad = 2.0 * math.atan(math.exp(y / RADIUS_EARTH)) - math.pi / 2.0

    # Convert latitude and longitude to degrees
    latitude = math.degrees(lat_rad)
    longitude = math.degrees(lon_rad)

    return latitude, -(90 + longitude)


def lat_lon_to_spherical_mercator(lat, lon):
    RADIUS_EARTH = 6378137.0  # Earth's radius in meters
    TILE_SIZE = 256.0  # Size of a tile in Web Mercator projection

    # Convert latitude and longitude to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Perform the conversion
    x = RADIUS_EARTH * lon_rad
    y = RADIUS_EARTH * math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0))

    # Scale and shift to fit within the Web Mercator coordinate system
    # x = (x + RADIUS_EARTH) / TILE_SIZE
    # y = (y + RADIUS_EARTH) / TILE_SIZE

    return x, y


map_size = (400, 400)


def get_static_map_image(lat, lon, zoom=15, size=map_size,
                         api_key='AIzaSyBI743mOAfTFDyuJt-cpTMAy-_58oq-vu8'):
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{size[0]}x{size[1]}",
        "maptype": "satellite",
        "key": api_key,
    }
    response = requests.get(base_url, params=params)
    return response.content


def calculate_bounding_box(center_lat, center_lon, zoom, map_size=map_size):
    # Constants for map dimensions at zoom level 0
    world_width = 256 * (2 ** zoom)
    world_height = 256 * (2 ** zoom)

    # Calculate pixel coordinates of the center
    center_x = world_width / 2
    center_y = world_height / 2

    # Calculate distance from center to the edge in pixels
    map_width, map_height = map_size
    edge_x = map_width / 2
    edge_y = map_height / 2

    # Calculate pixel coordinates of the corners
    top_left_x = center_x - edge_x
    top_left_y = center_y - edge_y
    bottom_right_x = center_x + edge_x
    bottom_right_y = center_y + edge_y

    # Convert pixel coordinates to latitude and longitude
    top_left_lat = center_lat + (90 - 360 * math.atan(math.exp(-(top_left_y / world_height - 0.5) * 2 * math.pi)) / math.pi)
    top_left_lon = center_lon + (top_left_x / world_width - 0.5) * 360

    bottom_right_lat = center_lat + (90 - 360 * math.atan(math.exp(-(bottom_right_y / world_height - 0.5) * 2 * math.pi)) / math.pi)
    bottom_right_lon = center_lon + (bottom_right_x / world_width - 0.5) * 360

    return top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon


def overlapped(label_a: tuple, label_b: tuple, overlap: int = 25):
    coords_a = calculate_bounding_box(label_a[0], label_a[1], label_a[2])
    coords_b = calculate_bounding_box(label_b[0], label_b[1], label_b[2])

    y_overlap = max(0, min(coords_a[2], coords_b[2]) - max(coords_a[0], coords_b[0]))
    x_overlap = max(0, min(coords_a[3], coords_b[3]) - max(coords_a[1], coords_b[1]))
    overlap_area = x_overlap * y_overlap
    rect1_area = (coords_a[2] - coords_a[0]) * (coords_a[3] - coords_a[1])
    rect2_area = (coords_b[2] - coords_b[0]) * (coords_b[3] - coords_b[1])
    overlap_percentage = (overlap_area / (rect1_area + rect2_area - overlap_area)) * 100
    if overlap_percentage >= overlap:
        return True
    return False
