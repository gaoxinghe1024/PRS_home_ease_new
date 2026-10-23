const assert=require('node:assert/strict');
const path=require('node:path');

// the translator reads window.HOMEEASE_EN; a language saved by an earlier version ('zh') must not bring Chinese back
global.window=global;
global.localStorage={getItem:()=>'zh',setItem(){}};
require(path.join(__dirname,'..','web','static','i18n-en.js'));
const {translate,api}=require(path.join(__dirname,'..','web','static','i18n.js'));

const CJK=/[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]/;
const table=global.HOMEEASE_EN;
const keys=Object.keys(table);

assert(keys.length>500,'the phrase table is loaded');
assert.equal(api.lang(),'en');

// every phrase of the table comes out in English
for(const k of keys){
  const out=translate(k);
  assert(!CJK.test(out),`still Chinese: ${JSON.stringify(k)} -> ${JSON.stringify(out)}`);
}

// text without Chinese is left alone
for(const s of ['Emergency','12:30','a 3 b','']) assert.equal(translate(s),s);

// a whole phrase is replaced by its entry; the surrounding blanks stay
const whole=keys.find(k=>k.length>12&&table[k].trim());
assert.equal(translate(` ${whole} `),` ${table[whole].trim()} `);

// two phrases joined by Chinese punctuation: English punctuation, one blank, no leftover Chinese
const [a,b]=keys.filter(k=>k.length>3&&table[k].trim()&&!CJK.test(table[k])).slice(0,2);
const joined=translate(`${a}\uff0c${b}\u3002`);
assert(!CJK.test(joined),joined);
assert(!/ {2}/.test(joined)&&!/\s[,.;:]/.test(joined),`spacing: ${JSON.stringify(joined)}`);
assert(joined.endsWith('.'),joined);

// two phrases side by side (no punctuation between them in Chinese) are two words; an ideographic space is a blank
assert.equal(translate('\u97f3\u9891\u53d7\u9650 / \u4e0d\u53ef\u7528'),'Audio limited / unavailable');
assert.equal(translate('5.6 min\u3000Audio'),'5.6 min Audio');
// a label that runs on into the next element keeps its blank after the colon
assert(translate('\u6388\u6743\u72b6\u6001\uff1a').endsWith(': '),translate('\u6388\u6743\u72b6\u6001\uff1a'));

// the language helper: Chinese in, English out; there is no Chinese mode and no switch any more
assert.equal(api.t(a),translate(a));
assert.equal(api.setLang,undefined);

console.log(`English interface: ${keys.length} phrases, spacing and English-only mode passed.`);
