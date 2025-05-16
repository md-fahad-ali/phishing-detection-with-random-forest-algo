async function testApi(url){
	console.log('Fetching URL:', url)
	  try {
	    const response = await fetch('http://localhost:5000/api/check', {
	      method: 'POST',
	      body: JSON.stringify({
	        url: url,
	      }),
	      headers: {
	        'Content-Type': 'application/json',
	      },
	    })
	    const data = await response.json()
	    console.log('Response from URL check:', data)
	    return data
	  } catch (error) {
	    console.error('Error checking URL:', error)
	    return { error: true }
	  }
}

testApi('https://google.com');
