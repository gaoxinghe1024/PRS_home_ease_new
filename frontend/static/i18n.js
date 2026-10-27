"use strict";
/* English interface for every page. The pages are written in Chinese; the text of the page (and what the scripts add later) is
   translated at the DOM level with the phrase table in i18n-en.js. The interface is English only: there is no language switch, and a
   language saved by an earlier version is ignored. care/i18n.py applies the same table and rules to what the server sends directly. */
(function (root) {
  var CJK = /[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]/;
  var EN = root.HOMEEASE_EN || {};
  var has = Object.prototype.hasOwnProperty;
  var lang = "en";

  var keys = Object.keys(EN).sort(function (a, b) { return b.length - a.length; });
  var rx = keys.length ? new RegExp(keys.map(function (k) { return k.replace(/[.*+?^${}()|[\]\\\/-]/g, "\\$&"); }).join("|"), "g") : null;
  var PUNCT = { "\uff0c": ", ", "\u3002": ". ", "\uff1a": ": ", "\uff1b": "; ", "\uff1f": "? ", "\uff01": "! ", "\uff08": " (", "\uff09": ") ",
                "\u3001": ", ", "\u300c": "\"", "\u300d": "\"", "\u201c": "\"", "\u201d": "\"", "\u3000": " " };
  var MARK_A = "\ue000", MARK_B = "\ue001";

  function translate(s) {
    if (typeof s !== "string" || !CJK.test(s)) return s;
    var lead = /^\s*/.exec(s)[0], trail = /\s*$/.exec(s)[0];
    var core = s.slice(lead.length, s.length - trail.length);
    // a text ending with a colon or comma runs on into the next element ("Status: <span>Off</span>"): keep its blank
    var gap = !trail && /[\uff1a\uff0c\uff1b\u3001]$/.test(core) ? " " : "";
    if (has.call(EN, core)) return lead + EN[core].replace(/^\s+|\s+$/g, "") + gap + trail;
    var out = rx ? core.replace(rx, function (m) { return MARK_A + EN[m] + MARK_B; }) : core;
    // spaces between an English phrase and a neighbouring letter or digit, or the next phrase
    out = out.replace(/([A-Za-z0-9.,;:!?)])\ue001\ue000(?=[A-Za-z0-9(])/g, "$1 \ue001\ue000");
    out = out.replace(/([A-Za-z0-9])\ue000(?=[A-Za-z0-9])/g, "$1 \ue000").replace(/([A-Za-z0-9.,;:!?)])\ue001(?=[A-Za-z0-9(])/g, "$1 ");
    // a phrase in the middle of a sentence starts with a small letter
    out = out.replace(/([a-z,;] ?)(\ue001?)\ue000([A-Z])(?=[a-z])/g, function (all, pre, mid, ch) { return pre + mid + "\ue000" + ch.toLowerCase(); });
    out = out.replace(/\ue000|\ue001/g, "");
    out = out.replace(/[\uff0c\u3002\uff1a\uff1b\uff1f\uff01\uff08\uff09\u3001\u300c\u300d\u201c\u201d\u3000]/g, function (c) { return PUNCT[c]; });
    out = out.replace(/ {2,}/g, " ").replace(/\s+([,.;:?!)])/g, "$1").replace(/\(\s+/g, "(");
    out = out.replace(/([,;:!?])(?=[A-Za-z])/g, "$1 ").replace(/(\.)(?=[A-Z])/g, "$1 ");
    return lead + out.replace(/^\s+|\s+$/g, "") + gap + trail;
  }

  var ATTRS = ["placeholder", "title", "aria-label", "alt"];
  var SKIP = { SCRIPT: 1, STYLE: 1, TEXTAREA: 1, NOSCRIPT: 1, CODE: 1 };

  function walk(node) {
    if (!node) return;
    if (node.nodeType === 3) {
      var p = node.parentNode;
      if (p && SKIP[p.nodeName]) return;
      var v = node.nodeValue;
      if (CJK.test(v)) { var t = translate(v); if (t !== v) node.nodeValue = t; }
      return;
    }
    if (node.nodeType !== 1) return;
    // the attributes are interface text also on a text box, whose content (what a person typed) is left alone
    for (var i = 0; i < ATTRS.length; i++) {
      var a = node.getAttribute(ATTRS[i]);
      if (a && CJK.test(a)) { var ta = translate(a); if (ta !== a) node.setAttribute(ATTRS[i], ta); }
    }
    if (SKIP[node.nodeName]) return;
    for (var c = node.firstChild; c; c = c.nextSibling) walk(c);
  }

  var api = { lang: function () { return lang; }, translate: translate, t: translate };
  root.HomeEaseI18n = api;
  if (typeof module !== "undefined" && module.exports) module.exports = { translate: translate, api: api };

  var d = root.document;
  if (!d || !d.documentElement) return;
  d.documentElement.lang = lang;
  ["toLocaleString", "toLocaleDateString", "toLocaleTimeString"].forEach(function (name) {
    var orig = Date.prototype[name];
    Date.prototype[name] = function (loc, opts) { return orig.call(this, loc === "zh-CN" ? "en-US" : loc, opts); };
  });
  // Native dialogs are not part of the page: their text is translated on the way in.
  ["alert", "confirm", "prompt"].forEach(function (name) {
    var orig = root[name];
    if (typeof orig !== "function") return;
    root[name] = function (message) {
      var args = Array.prototype.slice.call(arguments);
      args[0] = translate(message);
      return orig.apply(root, args);
    };
  });
  function start() {
    walk(d.documentElement);
    if (d.title) d.title = translate(d.title);
    if (typeof MutationObserver === "undefined") return;
    new MutationObserver(function (list) {
      for (var i = 0; i < list.length; i++) {
        var m = list[i];
        if (m.type === "childList") { for (var j = 0; j < m.addedNodes.length; j++) walk(m.addedNodes[j]); }
        else if (m.type === "characterData") walk(m.target);
        else if (m.type === "attributes" && m.target.nodeType === 1) {
          var a = m.target.getAttribute(m.attributeName);
          if (a && CJK.test(a)) { var ta = translate(a); if (ta !== a) m.target.setAttribute(m.attributeName, ta); }
        }
      }
    }).observe(d.documentElement, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ATTRS });
  }
  if (d.readyState === "loading") d.addEventListener("DOMContentLoaded", start); else start();
})(typeof window !== "undefined" ? window : globalThis);
