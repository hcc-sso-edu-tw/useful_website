var qrcode = new QRCode(document.getElementById("qrcode"), {
	text: "https://091cc.github.io/three-gay-kingdom/",
	width: 110,
	height: 110,
	colorDark : "#106faa",
	colorLight : "#ffffff",
	correctLevel : QRCode.CorrectLevel.H
});
