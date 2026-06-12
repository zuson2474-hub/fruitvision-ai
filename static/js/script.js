const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const loadingBox = document.getElementById("loadingBox");

imageInput.addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImage.src = e.target.result;
            previewImage.style.display = "block";
            previewImage.classList.add("zoomIn");
            loadingBox.style.display = "block";
        }
        reader.readAsDataURL(file);
    }
});