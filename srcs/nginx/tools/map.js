// Initialize map
var map = L.map('map').setView([48.8566, 2.3522], 2); // Default to Paris
var markers = [];

// Tile layer (OpenStreetMap)
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// Function to update the map with new suggestions
function updateMap(suggestions) {
	latitude = suggestions.latitude;
	longitude = suggestions.longitude;
	city = suggestions.city;
  let has_markers = false;
  for (const marker of markers) {
    map.removeLayer(marker);
  }
  if (latitude && longitude) {
		has_markers = true;
		var marker = L.marker([latitude, longitude]).addTo(map);
		markers.push(marker);
		// update map view
		map.setView([latitude, longitude], 8);
		// Add popup with city name
		// marker.bindPopup(city).openPopup();
		display_stat_informations(suggestions, marker);

	} 
	if (!has_markers) {
		alert('No suggestions found for your file. Please try again!');
	}
}

function display_stat_informations(suggestions, marker) {
	// Display statistics
	gradient_boosting_results = suggestions.gradient_boosting_results;
	knn_results = suggestions.knn_results;
	linear_gradient_results = suggestions.linear_gradient_results;
	random_forest_results = suggestions.random_forest_results;

	// Get MAE_TEST
	gradient_boosting_MAE = gradient_boosting_results.MAE_TEST;
	knn_boosting_MAE = knn_results.MAE_TEST;
	linear_gradient_MAE = linear_gradient_results.MAE_TEST;
	random_forest_MAE = random_forest_results.MAE_TEST;

	// Get R2_TEST
	gradient_boosting_R2 = gradient_boosting_results.R2_TEST;
	knn_R2 = knn_results.R2_TEST;
	linear_gradient_R2 = linear_gradient_results.R2_TEST;
	random_forest_R2 = random_forest_results.R2_TEST;

	// Get RMSE_TEST
	gradient_boosting_RMSE = gradient_boosting_results.RMSE_TEST;
	knn_RMSE = knn_results.RMSE_TEST;
	linear_gradient_RMSE = linear_gradient_results.RMSE_TEST;
	random_forest_RMSE = random_forest_results.RMSE_TEST;
	// Display statistics
	marker.bindPopup(
		'<div style="font-family: Arial, sans-serif; font-size: 14px;">' +
		'<h3 style="margin: 0; padding: 0;">' + suggestions.city + '</h3>' +
		'<hr style="margin: 5px 0;">' +
		'<b>Gradient Boosting:</b><br>' +
		'MAE: ' + gradient_boosting_MAE + '<br>' +
		'R2: ' + gradient_boosting_R2 + '<br>' +
		'RMSE: ' + gradient_boosting_RMSE + '<br>' +
		'<b>KNN:</b><br>' +
		'MAE: ' + knn_boosting_MAE + '<br>' +
		'R2: ' + knn_R2 + '<br>' +
		'RMSE: ' + knn_RMSE + '<br>' +
		'<b>Linear Gradient:</b><br>' +
		'MAE: ' + linear_gradient_MAE + '<br>' +
		'R2: ' + linear_gradient_R2 + '<br>' +
		'RMSE: ' + linear_gradient_RMSE + '<br>' +
		'<b>Random Forest:</b><br>' +
		'MAE: ' + random_forest_MAE + '<br>' +
		'R2: ' + random_forest_R2 + '<br>' +
		'RMSE: ' + random_forest_RMSE + '<br>' +
		'</div>'
	).openPopup();
}