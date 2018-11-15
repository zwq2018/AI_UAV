/*
Author: mg12
Author URI: http://www.neoease.com/
*/
(function() {

function $(id) {
	return document.getElementById(id);
}

function reply(authorId, commentId, commentBox) {
	var author = MGJS.$(authorId).innerHTML;
	var insertStr = '<a href="#' + commentId + '">@' + author.replace(/\t|\n|\r\n/g, "") + '</a> \n';

	appendReply(insertStr, commentBox);
}

function quote(authorId, commentId, commentBodyId, commentBox) {
	var author = MGJS.$(authorId).innerHTML;
	var comment = MGJS.$(commentBodyId).innerHTML;

	var insertStr = '<blockquote cite="#' + commentBodyId + '">';
	insertStr += '\n<a href="#' + commentId + '">' + author.replace(/\t|\n|\r\n/g, "") + '</a> :';
	insertStr += comment.replace(/\t/g, "");
	insertStr += '</blockquote>\n';

	insertQuote(insertStr, commentBox);
}

function appendReply(insertStr, commentBox) {
	if(MGJS.$(commentBox) && MGJS.$(commentBox).type == 'textarea') {
		field = MGJS.$(commentBox);

	} else {
		alert("The comment box does not exist!");
		return false;
	}

	if (field.value.indexOf(insertStr) > -1) {
		alert("You've already appended this reply!");
		return false;
	}

	if (field.value.replace(/\s|\t|\n/g, "") == '') {
		field.value = insertStr;
	} else {
		field.value = field.value.replace(/[\n]*$/g, "") + '\n\n' + insertStr;
	}
	field.focus();
}

function insertQuote(insertStr, commentBox) {
	if(MGJS.$(commentBox) && MGJS.$(commentBox).type == 'textarea') {
		field = MGJS.$(commentBox);

	} else {
		alert("The comment box does not exist!");
		return false;
	}

	if(document.selection) {
		field.focus();
		sel = document.selection.createRange();
		sel.text = insertStr;
		field.focus();

	} else if (field.selectionStart || field.selectionStart == '0') {
		var startPos = field.selectionStart;
		var endPos = field.selectionEnd;
		var cursorPos = startPos;
		field.value = field.value.substring(0, startPos)
					+ insertStr
					+ field.value.substring(endPos, field.value.length);
		cursorPos += insertStr.length;
		field.focus();
		field.selectionStart = cursorPos;
		field.selectionEnd = cursorPos;

	} else {
		field.value += insertStr;
		field.focus();
	}
}


window['MGJS'] = {};
window['MGJS']['$'] = $;
window['MGJS_CMT'] = {};
window['MGJS_CMT']['reply'] = reply;
window['MGJS_CMT']['quote'] = quote;

})();
