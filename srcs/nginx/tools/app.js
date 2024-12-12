// script.js

// DOM elements
const submitButton = document.getElementById('submit-button');
const travelInput = document.getElementById('city-select');
const mapContainer = document.getElementById('map');
// Backend API endpoint
const API_ENDPOINT = ' http://127.0.0.1:8888/api/postdata'; // Replace with your actual backend endpoint

// Function to handle form submission
function handleFormSubmission() {
  const userInput = travelInput.value.trim();
  
  if (!userInput) {
    alert('Please enter a destination or preference!');
    return;
  }

  // Add loading state to the submit button
  submitButton.disabled = true;
  submitButton.innerHTML = 'Loading...';

  // Send the user input to the backend API
  fetch(API_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ city: userInput }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }
      submitButton.disabled = false;
      submitButton.innerHTML = 'Submit';
      return response.json();
    })
    .then((data) => {
      submitButton.disabled = false;
      submitButton.innerHTML = 'Submit';
      updateMap(data);
			// Show data in the html page
			// document.getElementById('suggestions').innerHTML = data.suggestions;


    })
    .catch((error) => {
      submitButton.disabled = false;
      submitButton.innerHTML = 'Submit';
      console.error('Error fetching suggestions:', error);
      alert('Failed to fetch suggestions. Please try again!');
    });
}

// Attach event listener to the submit button
submitButton.addEventListener('click', handleFormSubmission);
