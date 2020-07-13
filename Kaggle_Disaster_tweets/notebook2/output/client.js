let axios = require('axios');
axios.post('http://127.0.0.1:5000/', {
  mytext: 'It is raining too much',
})
.then((response) => {
  console.log(response.data);
}, (error) => {
  console.log(error);
});
