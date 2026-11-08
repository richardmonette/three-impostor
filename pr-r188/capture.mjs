// Captures the images for the three.js PR from a local three.js checkout.
//
//   cd path/to/three.js && npx servez -p 8080 .   (or any static server)
//   THREE_JS=path/to/three.js node capture.mjs out/
//
// Writes row.png (the default five dragon LOD row) and grid.png (400 dragons),
// with the info text and settings hidden. Runs a visible browser so the real GPU is used.

import { createRequire } from 'module';

const require = createRequire( ( process.env.THREE_JS || '../three.js' ) + '/package.json' );
const puppeteer = require( 'puppeteer' );

const out = process.argv[ 2 ] || '.';
const url = process.env.URL || 'http://localhost:8080/examples/webgpu_impostors.html';

const browser = await puppeteer.launch( { headless: false, args: [ '--hide-scrollbars', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist' ] } );
const page = await browser.newPage();
await page.setViewport( { width: 1000, height: 560, deviceScaleFactor: 2 } );
page.on( 'pageerror', ( e ) => console.log( 'pageerror:', e.message ) );

await page.goto( url, { waitUntil: 'domcontentloaded' } );
await page.waitForFunction( () => document.querySelectorAll( '.label' ).length === 5 && document.querySelector( '.label' ).textContent, { timeout: 90000 } );

// keep the canvas and the labels only
await page.evaluate( () => {

	for ( const el of document.body.children ) {

		if ( el.tagName !== 'CANVAS' && el.tagName !== 'SCRIPT' && ! el.querySelector( '.label' ) ) el.style.visibility = 'hidden';

	}

} );

const wait = ( ms ) => new Promise( ( r ) => setTimeout( r, ms ) );

await wait( 2500 );
await page.screenshot( { path: `${ out }/row.png` } );

// the first select in the settings is the dragon count
await page.evaluate( () => {

	const select = document.querySelectorAll( 'select' )[ 0 ];
	select.value = '400';
	select.dispatchEvent( new Event( 'change', { bubbles: true } ) );

} );

await wait( 5000 );
await page.screenshot( { path: `${ out }/grid.png` } );

await browser.close();
