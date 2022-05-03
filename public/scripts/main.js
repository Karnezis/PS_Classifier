function init() {
  url = "http://pathospotter.uefs.br/ps-api/";
  axios.get(url).then((response) => console.log(response.data));
  sendImage();
}

function sendImage() {
  window.onload = function () {
    const form = document.getElementById("submit-image");
    form.onsubmit = async (event) => {
      event.preventDefault();
      let formData = new FormData();
      formData.append("file", document.getElementById("img").files[0]);
      let submit = url + "api/predict";
      let response = await axios.post("http://pathospotter.uefs.br/ps-api/api/predict", formData);
      await console.log(response.data);
    };
  };
}

init();
