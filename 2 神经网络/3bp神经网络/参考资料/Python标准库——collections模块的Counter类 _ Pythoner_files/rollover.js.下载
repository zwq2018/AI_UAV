/**
 * jQuery.rollover
 *
 * @version  1.0.2
 * @author   rew <rewish.org@gmail.com>
 * @link     http://rewish.org/javascript/jquery_rollover_plugin
 * @license  http://rewish.org/license/mit The MIT License
 *
 * Inspired by:
 * Telepath Labs (http://dev.telepath.co.jp/labs/article.php?id=15)
 *
 */
jQuery.fn.rollover = function(suffix) {
	suffix = suffix || '_on';
	return this.not('[src*="'+ suffix +'."]').each(function() {
		var img = jQuery(this);
		var src = img.attr('src');
		var _on = [
			src.substr(0, src.lastIndexOf('.')),
			src.substring(src.lastIndexOf('.'))
		].join(suffix);
		jQuery('<img>').attr('src', _on);
		img.hover(
			function() { img.attr('src', _on); },
			function() { img.attr('src', src); }
		);
	});
};