window.addEventListener('load', () => {
	const canvas = document.querySelector("#canvas")  // Upper canvas for drawing
	const canvas2 = document.querySelector("#canvas2")  // Lower canvas for background image
	const ctx = canvas.getContext("2d")
	const ctx2 = canvas2.getContext("2d")
	var slider = document.getElementById("myRange")
	var srcImg = document.getElementById("srcImg")
	var button = document.getElementById("inpaint")
	var button_download = document.getElementById("download")
	let painting = false

	function resizeCanvas(height, width){
		canvas.height = height
		canvas.width = width
		canvas2.height = height
		canvas2.width = width
	}

	function startPosition(e){
		painting = true
		draw(e)
	}

	function finishedPosition(){
		painting = false
		ctx.beginPath()
	}

	function draw(e){
		if(!painting) return
		ctx.lineWidth = slider.value   // Change brush size based on slider value
		ctx.lineCap = "round"
		ctx.lineTo(e.pageX - 8, e.pageY - 326)
		ctx.stroke()
		ctx.beginPath()
		ctx.moveTo(e.pageX - 8, e.pageY - 326)
	}

	function make_base(){
		img = new Image()
		img.src = srcImg.src
		img.onload = function(){
			resizeCanvas(img.height, img.width)
			ctx2.drawImage(img, 0, 0)
		}
		console.log('make base')
	}

	function inpaint(){
		var mask = canvas.toDataURL()
		var new_img = document.getElementById("srcImg").getAttribute('src')
		async function getInpaintedImage() {
			const response = await fetch('/result/');
			const data = await response.json();
			console.log('inpainted')
			$("#srcImg").attr("src", data["img"]);
			make_base()
		}

		jQuery.ajax({
      		type: "POST",
      		url: "/register/",
      		data: { 
      			mask: mask,
      			new_img: new_img
      		},
      		success: function(data) {
      			console.log("canvas saved")
      			getInpaintedImage();
      		}
    	})
	}

	function download_img(e){
		e.preventDefault()
		const downloadLink = document.createElement("a")
	    downloadLink.download = "New_image"
	    downloadLink.innerHTML = "Download File"
	    downloadLink.href = srcImg.src
	    downloadLink.click()
	}

	// Event listeners
	button_download.addEventListener("click", download_img)
	button.addEventListener("click", inpaint)
	canvas.addEventListener("mousedown", startPosition)
	canvas.addEventListener("mouseup", finishedPosition)
	canvas.addEventListener("mousemove", draw)

	make_base()
});