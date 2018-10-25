var board = document.getElementsByClassName("board")
var dict = {'blue': [], 'red': [], 'assassin': [], 'neutral': []}
for(var i = 0; i < board[0].children.length; i++){
	let child = board[0].children[i]
	let word = child.firstElementChild
	if(child.classList.contains('blue')){
		dict['blue'].push(word.innerText.toLowerCase())
	} else if (child.classList.contains('red')){
		dict['red'].push(word.innerText.toLowerCase())
	} else if (child.classList.contains('neutral')) {
		dict['neutral'].push(word.innerText.toLowerCase())
	} else {
		dict['assassin'].push(word.innerText.toLowerCase())
	}
}
console.log(dict)