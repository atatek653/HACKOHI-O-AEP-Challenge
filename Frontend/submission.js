// script.js
document.querySelector('request-comment').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form from submitting normally


    const date = document.querySelector('#date').value;
    const comment = document.querySelector('#comment').value;
    
   

    const submission = {
        date: date,
        comment: comment
    };

    // Send data to server
    fetch('/addComment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(submission)
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});
