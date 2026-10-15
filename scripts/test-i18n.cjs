const assert=require('node:assert/strict');
const path=require('node:path');

// the translator reads window.HOMEEASE_EN and the saved language; English mode is selected before it is loaded
global.window=global;
global.localStorage={getItem:()=>'en',setItem(){}};
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

// the language helper: Chinese in, English out
assert.equal(api.t(a),translate(a));

// Chinese mode does nothing at all
delete require.cache[require.resolve(path.join(__dirname,'..','web','static','i18n.js'))];
global.localStorage={getItem:()=>'zh',setItem(){}};
const zh=require(path.join(__dirname,'..','web','static','i18n.js')).api;
assert.equal(zh.lang(),'zh');
assert.equal(zh.t(a),a);

console.log(`Language switch: ${keys.length} phrases, spacing and Chinese mode passed.`);
