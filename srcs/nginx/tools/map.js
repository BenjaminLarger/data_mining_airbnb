// Initialize map
var map = L.map('map').setView([48.8566, 2.3522], 2); // Default to Paris
var markers = [];

// Tile layer (OpenStreetMap)
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// Function to update the map with new suggestions
function updateMap(suggestions) {
  let has_markers = false;
  for (const marker of markers) {
    map.removeLayer(marker);
  }
  console.log('Updating map with new suggestions:', suggestions);
  for (const [city, coordinates] of Object.entries(suggestions)) {
    if (coordinates && coordinates.latitude && coordinates.longitude) {
      has_markers = true;
      console.log(`City: ${city}`);
      console.log(`Latitude: ${coordinates.latitude}`);
      console.log(`Longitude: ${coordinates.longitude}`);
      var marker = L.marker([coordinates.latitude, coordinates.longitude]).addTo(map);
      markers.push(marker);
    } else {
      console.log('Invalid city coordinates:', city, coordinates);
    }
    if (!has_markers) {
      alert('No suggestions found for your file. Please try again!');
    }
  }
}