let currentIndex = 0;
let data = [];

// Fetch data from the JSON file
fetch('data.json')
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(fetchedData => {
        data = fetchedData;
        updateDisplay(); // Update display after data is fetched
    })
    .catch(error => console.error('Error fetching data:', error));

// Function to update the display
function updateDisplay() {
    if (data.length === 0) return; // Check if data is available
    const commentDiv = document.querySelector(".comment p"); // Select the paragraph inside the comment cell
    const matplotlibImg = document.querySelector(".matplotlib");
    const embeddingsImg = document.querySelector(".embeddings");

    // Update comment
    commentDiv.textContent = data[currentIndex].comment; // Display the comment
    // Update plot images (assuming your JSON has appropriate image URLs)
    matplotlibImg.src = data[currentIndex].matplotlibPlot; // Image URL for matplotlib
    embeddingsImg.src = data[currentIndex].embeddingsPlot; // Image URL for embeddings
}

// Event listener for the button
document.querySelector(".next-button").addEventListener("click", () => {
    currentIndex++;
    if (currentIndex >= data.length) {
        currentIndex = 0; // Loop back to the start
    }
    updateDisplay();
});
