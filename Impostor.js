import {
	BackSide,
	Box3,
	BoxGeometry,
	Color,
	DoubleSide,
	DynamicDrawUsage,
	HalfFloatType,
	InstancedInterleavedBuffer,
	Matrix4,
	Mesh,
	MeshStandardNodeMaterial,
	NodeMaterial,
	OrthographicCamera,
	PlaneGeometry,
	RenderTarget,
	Scene,
	Vector3
} from 'three/webgpu';
import {
	Break,
	Discard,
	Fn,
	If,
	Loop,
	OnBeforeFrameUpdate,
	abs,
	bool,
	buffer,
	cameraFar,
	cameraNear,
	cameraPosition,
	cameraProjectionMatrix,
	cameraViewMatrix,
	cameraWorldMatrix,
	ceil,
	clamp,
	cross,
	depth,
	diffuseColor,
	dot,
	exp,
	float,
	floor,
	instanceIndex,
	instancedBufferAttribute,
	instancedDynamicBufferAttribute,
	int,
	inverse,
	length,
	mat3,
	mat4,
	materialColor,
	max,
	min,
	modelWorldMatrix,
	mrt,
	normalView,
	normalize,
	positionGeometry,
	positionWorld,
	property,
	select,
	storage,
	texture,
	transpose,
	uniform,
	uv,
	varying,
	vec2,
	vec3,
	vec4,
	viewZToLogarithmicDepth,
	viewZToOrthographicDepth,
	viewZToPerspectiveDepth,
	viewZToReversedOrthographicDepth,
	viewZToReversedPerspectiveDepth
} from 'three/tsl';

// Box layout: six views, each looking into the box from one side. `width` and `height` are the
// box axes that map to the image's horizontal and vertical directions.
//
//   view    image u    image v    depth from that side
//   +X      1 - z      y          1 - x
//   -X      z          y          x
//   +Y      x          1 - z      1 - y
//   -Y      x          z          y
//   +Z      x          y          1 - z
//   -Z      1 - x      y          z

const VIEWS = [
	{ side: [ 1, 0, 0 ], up: [ 0, 1, 0 ], axis: 0, width: 2, height: 1 },
	{ side: [ - 1, 0, 0 ], up: [ 0, 1, 0 ], axis: 0, width: 2, height: 1 },
	{ side: [ 0, 1, 0 ], up: [ 0, 0, - 1 ], axis: 1, width: 0, height: 2 },
	{ side: [ 0, - 1, 0 ], up: [ 0, 0, 1 ], axis: 1, width: 0, height: 2 },
	{ side: [ 0, 0, 1 ], up: [ 0, 1, 0 ], axis: 2, width: 0, height: 1 },
	{ side: [ 0, 0, - 1 ], up: [ 0, 1, 0 ], axis: 2, width: 0, height: 1 }
];

const viewUV = [
	( b ) => vec2( b.z.oneMinus(), b.y ),
	( b ) => vec2( b.z, b.y ),
	( b ) => vec2( b.x, b.z.oneMinus() ),
	( b ) => vec2( b.x, b.z ),
	( b ) => vec2( b.x, b.y ),
	( b ) => vec2( b.x.oneMinus(), b.y )
];

// The interval texture has one tile per axis, addressed by the two other coordinates.

const axisUV = [
	( b ) => vec2( b.z, b.y ),
	( b ) => vec2( b.x, b.z ),
	( b ) => vec2( b.x, b.y )
];

// Octahedral layout: an N x N grid of views over the sphere (or the upper hemisphere), placed with
// the octahedral mapping so neighbouring grid points are neighbouring directions. Each view looks
// at the centre from that direction. The same up rule is used when baking and when shading.

function octahedralDirection( u, v, hemisphere, target ) {

	const px = u * 2 - 1, py = v * 2 - 1;

	if ( hemisphere ) {

		const x = ( px + py ) / 2, z = ( px - py ) / 2;
		return target.set( x, 1 - Math.abs( x ) - Math.abs( z ), z ).normalize();

	}

	let x = px, z = py;
	const y = 1 - Math.abs( x ) - Math.abs( z );

	if ( y < 0 ) {

		const fx = ( 1 - Math.abs( z ) ) * ( x >= 0 ? 1 : - 1 );
		const fz = ( 1 - Math.abs( x ) ) * ( z >= 0 ? 1 : - 1 );
		x = fx;
		z = fz;

	}

	return target.set( x, y, z ).normalize();

}

const viewUp = ( direction ) => Math.abs( direction.y ) > 0.999 ? [ 0, 0, - 1 ] : [ 0, 1, 0 ];

const signNotZero = ( v ) => vec2( select( v.x.greaterThanEqual( 0 ), 1, - 1 ), select( v.y.greaterThanEqual( 0 ), 1, - 1 ) );

function octahedralEncode( n, hemisphere ) {

	if ( hemisphere ) {

		const h = vec3( n.x, max( n.y, 0 ), n.z ).toVar();
		const p = h.xz.div( abs( h.x ).add( h.y ).add( abs( h.z ) ) ).toVar();
		return vec2( p.x.add( p.y ), p.x.sub( p.y ) ).mul( 0.5 ).add( 0.5 );

	}

	const p = n.xz.div( abs( n.x ).add( abs( n.y ) ).add( abs( n.z ) ) ).toVar();
	const folded = vec2( 1 ).sub( abs( p.yx ) ).mul( signNotZero( p ) );
	return select( n.y.lessThan( 0 ), folded, p ).mul( 0.5 ).add( 0.5 );

}

function octahedralDecode( grid, hemisphere ) {

	const p = grid.mul( 2 ).sub( 1 ).toVar();

	if ( hemisphere ) {

		const x = p.x.add( p.y ).mul( 0.5 ).toVar(), z = p.x.sub( p.y ).mul( 0.5 ).toVar();
		return normalize( vec3( x, abs( x ).add( abs( z ) ).oneMinus(), z ) );

	}

	const n = vec3( p.x, abs( p.x ).add( abs( p.y ) ).oneMinus(), p.y ).toVar();
	const folded = vec2( 1 ).sub( abs( n.zx ) ).mul( signNotZero( n.xz ) ).toVar();
	const below = n.y.lessThan( 0 );
	return normalize( vec3( select( below, folded.x, n.x ), n.y, select( below, folded.y, n.z ) ) );

}

const _center = /*@__PURE__*/ new Vector3();
const _size = /*@__PURE__*/ new Vector3();
const _direction = /*@__PURE__*/ new Vector3();
const _tile = /*@__PURE__*/ new Matrix4();
const _clearColor = /*@__PURE__*/ new Color();

/**
 * The result of {@link ImpostorBaker#bake}: the textures that describe an object, plus the box they
 * were captured in. Pass it to {@link ImpostorNodeMaterial} and draw it with {@link ImpostorData#geometry}.
 */
class ImpostorData {

	constructor( { layout, box, resolution, frames = 0, hemisphere = false, radius = 0, colorTarget, normalTarget, intervalTarget = null } ) {

		this.isImpostorData = true;

		/**
		 * How the object was captured: `'box'` or `'octahedral'`.
		 *
		 * @type {string}
		 */
		this.layout = layout;

		/**
		 * The box the object was captured in, in the object's local space.
		 *
		 * @type {Box3}
		 */
		this.box = box;

		/**
		 * The size in pixels of each captured view.
		 *
		 * @type {number}
		 */
		this.resolution = resolution;

		/**
		 * Octahedral layout: the number of views along each side of the grid.
		 *
		 * @type {number}
		 */
		this.frames = frames;

		/**
		 * Octahedral layout: whether the views only cover the upper hemisphere.
		 *
		 * @type {boolean}
		 */
		this.hemisphere = hemisphere;

		/**
		 * Octahedral layout: half the width of each view, around the centre of the box.
		 *
		 * @type {number}
		 */
		this.radius = radius;

		/**
		 * Base colour of the views. Box layout: 3 x 2 grid. Octahedral layout: frames x frames grid.
		 *
		 * @type {Texture}
		 */
		this.colorMap = colorTarget.texture;

		/**
		 * Object-space normals of the views, in the same grid. For the octahedral layout, alpha holds
		 * the height of the surface towards the view.
		 *
		 * @type {Texture}
		 */
		this.normalMap = normalTarget.texture;

		/**
		 * Box layout: for each axis, the nearest surface seen from the positive side (red) and from
		 * the negative side (green, stored as one minus the coordinate), in a 3 x 1 grid.
		 *
		 * @type {?Texture}
		 */
		this.intervalMap = intervalTarget ? intervalTarget.texture : null;

		/**
		 * A box matching {@link ImpostorData#box}, to render the impostor with.
		 *
		 * @type {BoxGeometry}
		 */
		this.geometry = new BoxGeometry( ...box.getSize( _size ).toArray() ).translate( ...box.getCenter( _center ).toArray() );

		this._targets = [ colorTarget, normalTarget, intervalTarget ].filter( Boolean );

	}

	dispose() {

		for ( const target of this._targets ) target.dispose();
		this.geometry.dispose();

	}

}

/**
 * Captures an object so it can be drawn as an impostor with {@link ImpostorNodeMaterial}.
 *
 * ```js
 * const impostor = await new ImpostorBaker( renderer ).bake( mesh );
 * const lod = new THREE.LOD();
 * lod.addLevel( mesh, 0 );
 * lod.addLevel( new THREE.Mesh( impostor.geometry, new ImpostorNodeMaterial( impostor ) ), 20 );
 * ```
 */
class ImpostorBaker {

	/**
	 * @param {WebGPURenderer} renderer - The renderer used for capturing.
	 */
	constructor( renderer ) {

		this.renderer = renderer;

	}

	/**
	 * Captures the given object in its local space. Its own position, rotation and scale are ignored,
	 * so give the impostor the same transform as the object.
	 *
	 * The `'box'` layout captures six views and ray marches them, which gives true depth and holds
	 * up at closer range. The `'octahedral'` layout captures a grid of views and blends the three
	 * nearest, which is cheaper per pixel and suits objects further away.
	 *
	 * @param {Object3D} object - The object to capture.
	 * @param {Object} [options]
	 * @param {string} [options.layout='box'] - `'box'` or `'octahedral'`.
	 * @param {number} [options.resolution] - The size in pixels of each view: 256 for box, 128 for octahedral.
	 * @param {number} [options.frames=12] - Octahedral layout: views along each side of the grid.
	 * @param {boolean} [options.hemisphere=false] - Octahedral layout: only capture views from above.
	 * @return {Promise<ImpostorData>}
	 */
	async bake( object, { layout = 'box', resolution = layout === 'octahedral' ? 128 : 256, frames = 12, hemisphere = false } = {} ) {

		const renderer = this.renderer;
		await renderer.init();

		const source = object.clone();
		source.position.set( 0, 0, 0 );
		source.quaternion.identity();
		source.scale.set( 1, 1, 1 );
		source.updateMatrixWorld( true );

		const scene = new Scene();
		scene.add( source );

		// pad the box so no surface lies exactly on one of its sides

		const box = new Box3().setFromObject( source, true );
		const extent = box.getSize( _size ).toArray().reduce( ( a, b ) => Math.max( a, b ) );
		box.expandByScalar( extent * 0.02 );

		const previousTarget = renderer.getRenderTarget();
		const previousMRT = renderer.getMRT();
		const previousAutoClear = renderer.autoClear;
		const previousClearAlpha = renderer.getClearAlpha();
		renderer.getClearColor( _clearColor );

		renderer.setClearColor( 0x000000, 0 );

		const data = layout === 'octahedral'
			? this._bakeOctahedral( scene, box, resolution, frames, hemisphere )
			: this._bakeBox( scene, box, resolution );

		renderer.setRenderTarget( previousTarget );
		renderer.setMRT( previousMRT );
		renderer.autoClear = previousAutoClear;
		renderer.setClearColor( _clearColor, previousClearAlpha );

		return data;

	}

	_bakeBox( scene, box, resolution ) {

		const renderer = this.renderer;
		const size = box.getSize( new Vector3() );
		const center = box.getCenter( new Vector3() );
		const extent = Math.max( size.x, size.y, size.z );

		// capture the six views: base colour, normal, and the surface position in box space

		const boxPosition = positionWorld.sub( vec3( ...box.min.toArray() ) ).div( vec3( ...size.toArray() ) );

		const views = VIEWS.map( () => {

			const target = new RenderTarget( resolution, resolution, { count: 3, type: HalfFloatType } );
			target.textures[ 0 ].name = 'output';
			target.textures[ 1 ].name = 'normal';
			target.textures[ 2 ].name = 'position';
			return target;

		} );

		renderer.autoClear = true;
		renderer.setMRT( mrt( {
			output: vec4( diffuseColor.rgb, 1 ),
			normal: vec4( worldNormal().mul( 0.5 ).add( 0.5 ), 1 ),
			position: vec4( boxPosition, 1 )
		} ) );

		const camera = new OrthographicCamera();
		const margin = extent * 0.1;

		VIEWS.forEach( ( view, i ) => {

			const half = size.getComponent( view.axis ) / 2;
			const width = size.getComponent( view.width ) / 2;
			const height = size.getComponent( view.height ) / 2;

			camera.left = - width;
			camera.right = width;
			camera.top = height;
			camera.bottom = - height;
			camera.near = margin / 2;
			camera.far = margin + half * 2 + margin / 2;
			camera.position.copy( center ).addScaledVector( _direction.set( ...view.side ), half + margin );
			camera.up.set( ...view.up );
			camera.lookAt( center );
			camera.updateProjectionMatrix();
			camera.updateMatrixWorld();

			renderer.setRenderTarget( views[ i ] );
			renderer.render( scene, camera );

		} );

		renderer.setMRT( null );

		// pack the views into the colour and normal atlases, and opposite views into the interval atlas

		// render targets are sampled top row first, so flip v to read a capture in image space (v up)
		const capture = ( map, imageUV ) => texture( map, vec2( imageUV.x, imageUV.y.oneMinus() ) );

		const colorTarget = atlasTarget( 3 * resolution, 2 * resolution );
		const normalTarget = atlasTarget( 3 * resolution, 2 * resolution );
		const intervalTarget = atlasTarget( 3 * resolution, resolution );

		pack( renderer, colorTarget, views.map( ( view, i ) => ( { vertex: tile( i % 3, Math.floor( i / 3 ), 3, 2 ), fragment: capture( view.textures[ 0 ], uv() ) } ) ) );
		pack( renderer, normalTarget, views.map( ( view, i ) => ( { vertex: tile( i % 3, Math.floor( i / 3 ), 3, 2 ), fragment: capture( view.textures[ 1 ], uv() ).mul( capture( view.textures[ 0 ], uv() ).a ) } ) ) );

		const st = uv();
		const flipU = vec2( st.x.oneMinus(), st.y );
		const flipV = vec2( st.x, st.y.oneMinus() );
		const positiveUV = [ flipU, flipV, st ];
		const negativeUV = [ st, st, flipU ];

		pack( renderer, intervalTarget, [ 0, 1, 2 ].map( ( axis ) => {

			const positive = views[ axis * 2 ], negative = views[ axis * 2 + 1 ];
			const positiveCovered = capture( positive.textures[ 0 ], positiveUV[ axis ] ).a.greaterThan( 0.5 );
			const negativeCovered = capture( negative.textures[ 0 ], negativeUV[ axis ] ).a.greaterThan( 0.5 );
			const high = capture( positive.textures[ 2 ], positiveUV[ axis ] ).element( axis );
			const low = capture( negative.textures[ 2 ], negativeUV[ axis ] ).element( axis );

			return {
				vertex: tile( axis, 0, 3, 1 ),
				fragment: vec4( select( positiveCovered, high, 0 ), select( negativeCovered, low.oneMinus(), 0 ), 0, 1 )
			};

		} ) );

		for ( const view of views ) view.dispose();

		return new ImpostorData( { layout: 'box', box, resolution, colorTarget, normalTarget, intervalTarget } );

	}

	_bakeOctahedral( scene, box, resolution, frames, hemisphere ) {

		const renderer = this.renderer;
		const center = box.getCenter( new Vector3() );
		const radius = box.getSize( _size ).length() / 2;
		const margin = radius * 0.1;
		const size = frames * resolution;

		// capture every view straight into one atlas: base colour, normal, and the surface height
		// towards the view, measured from the plane through the centre

		const capture = new RenderTarget( size, size, { count: 2, type: HalfFloatType } );
		capture.textures[ 0 ].name = 'output';
		capture.textures[ 1 ].name = 'normal';

		const towardView = normalize( cameraWorldMatrix.element( 2 ).xyz );
		const height = dot( positionWorld.sub( vec3( ...center.toArray() ) ), towardView ).div( radius * 2 ).add( 0.5 );

		renderer.setMRT( mrt( {
			output: vec4( diffuseColor.rgb, 1 ),
			normal: vec4( worldNormal().mul( 0.5 ).add( 0.5 ), height )
		} ) );

		renderer.setRenderTarget( capture );
		renderer.clear();
		renderer.autoClear = false;

		const camera = new OrthographicCamera( - radius, radius, radius, - radius, margin / 2, margin + radius * 2 + margin / 2 );
		camera.coordinateSystem = renderer.coordinateSystem;

		for ( let j = 0; j < frames; j ++ ) {

			for ( let i = 0; i < frames; i ++ ) {

				octahedralDirection( i / ( frames - 1 ), j / ( frames - 1 ), hemisphere, _direction );

				camera.position.copy( center ).addScaledVector( _direction, radius + margin );
				camera.up.set( ...viewUp( _direction ) );
				camera.lookAt( center );
				camera.updateMatrixWorld();
				camera.updateProjectionMatrix();

				// squeeze the view into its tile; rows count down from the top, as render targets are sampled
				_tile.set(
					1 / frames, 0, 0, ( 1 + 2 * i ) / frames - 1,
					0, 1 / frames, 0, 1 - ( 1 + 2 * j ) / frames,
					0, 0, 1, 0,
					0, 0, 0, 1
				);
				camera.projectionMatrix.premultiply( _tile );
				camera.projectionMatrixInverse.copy( camera.projectionMatrix ).invert();

				renderer.render( scene, camera );

			}

		}

		renderer.setMRT( null );
		renderer.autoClear = true;

		// copy out, masking normal and height by coverage so filtering at silhouettes can be undone

		const colorTarget = atlasTarget( size, size );
		const normalTarget = atlasTarget( size, size );
		const whole = tile( 0, 0, 1, 1 );

		pack( renderer, colorTarget, [ { vertex: whole, fragment: texture( capture.textures[ 0 ], uv() ) } ] );
		pack( renderer, normalTarget, [ { vertex: whole, fragment: texture( capture.textures[ 1 ], uv() ).mul( texture( capture.textures[ 0 ], uv() ).a ) } ] );

		capture.dispose();

		return new ImpostorData( { layout: 'octahedral', box, resolution, frames, hemisphere, radius, colorTarget, normalTarget } );

	}

}

const worldNormal = () => normalize( cameraWorldMatrix.mul( vec4( normalView, 0 ) ).xyz );

const atlasTarget = ( width, height ) => new RenderTarget( width, height, { type: HalfFloatType, depthBuffer: false } );

// render targets are sampled top row first, so tiles are written upside down
function tile( column, row, columns, rows ) {

	const p = vec2( column, row ).add( uv() ).div( vec2( columns, rows ) ).mul( 2 ).sub( 1 );
	return vec4( p.x, p.y.negate(), 0, 1 );

}

const _quad = /*@__PURE__*/ new PlaneGeometry();
const _packCamera = /*@__PURE__*/ new OrthographicCamera();

function pack( renderer, target, quads ) {

	const packScene = new Scene();

	for ( const { vertex, fragment } of quads ) {

		const material = new NodeMaterial();
		material.side = DoubleSide; // the flipped tile reverses the winding
		material.depthTest = false;
		material.depthWrite = false;
		material.vertexNode = vertex;
		material.fragmentNode = fragment;

		const mesh = new Mesh( _quad, material );
		mesh.frustumCulled = false;
		packScene.add( mesh );

	}

	renderer.setRenderTarget( target );
	renderer.render( packScene, _packCamera );

	packScene.traverse( ( child ) => child.material?.dispose() );

}

// Instanced meshes: rebuild the per-instance matrix the same way three.js does, since the
// impostor needs it (and its inverse) to bring the camera into the object's space.

const _interleavedMatrices = /*@__PURE__*/ new WeakMap();

function instanceMatrixNode( builder, matrices ) {

	const count = Math.max( matrices.count, 1 );

	if ( matrices.isStorageInstancedBufferAttribute === true ) {

		return storage( matrices, 'mat4', count ).element( instanceIndex );

	}

	if ( count * 64 <= builder.getUniformBufferLimit() ) {

		return buffer( matrices.array, 'mat4', count ).element( instanceIndex );

	}

	let interleaved = _interleavedMatrices.get( matrices );

	if ( interleaved === undefined ) {

		interleaved = new InstancedInterleavedBuffer( matrices.array, 16, 1 );
		_interleavedMatrices.set( matrices, interleaved );

	}

	OnBeforeFrameUpdate( () => {

		if ( interleaved.version !== matrices.version ) interleaved.version = matrices.version;

	} );

	const attribute = matrices.usage === DynamicDrawUsage ? instancedDynamicBufferAttribute : instancedBufferAttribute;

	return mat4(
		attribute( interleaved, 'vec4', 16, 0 ),
		attribute( interleaved, 'vec4', 16, 4 ),
		attribute( interleaved, 'vec4', 16, 8 ),
		attribute( interleaved, 'vec4', 16, 12 )
	);

}

/**
 * Draws an object captured by {@link ImpostorBaker}. Use it with {@link ImpostorData#geometry}
 * on a `Mesh` or `InstancedMesh`.
 *
 * Box layout: each pixel marches a ray through the captured box. A point is solid when it lies
 * between the nearest surfaces seen from both sides of every axis, which costs up to three texture
 * reads per step.
 *
 * Octahedral layout: each pixel blends the three captured views nearest the view direction, each
 * shifted along the ray by its stored height.
 *
 * Either way the result is shaded with the captured colour and normal through the usual three.js
 * lighting, and writes its depth. Shadows are not supported yet.
 */
class ImpostorNodeMaterial extends MeshStandardNodeMaterial {

	static get type() {

		return 'ImpostorNodeMaterial';

	}

	/**
	 * @param {ImpostorData} impostor - The data returned by {@link ImpostorBaker#bake}.
	 * @param {Object} [parameters] - Material parameters, as for `MeshStandardNodeMaterial`.
	 */
	constructor( impostor, parameters ) {

		super();

		this.isImpostorNodeMaterial = true;

		/**
		 * @type {ImpostorData}
		 */
		this.impostor = impostor;

		/**
		 * Box layout: uniform holding the largest number of ray steps for a pixel.
		 *
		 * @type {UniformNode<float>}
		 */
		this.stepsNode = uniform( 64 );

		this.side = BackSide; // back faces exist whether the camera is outside or inside the box
		this.roughness = 1;
		this.metalness = 0;

		this.setValues( parameters );

	}

	/**
	 * Box layout: the largest number of ray steps for a pixel.
	 *
	 * @type {number}
	 * @default 64
	 */
	get steps() {

		return this.stepsNode.value;

	}

	set steps( value ) {

		this.stepsNode.value = value;

	}

	setupDiffuseColor( builder ) {

		// the impostor is resolved as part of the base colour, which three.js builds before the
		// normal is needed; it also produces the normal and the depth of the hit
		const normal = property( 'vec3', 'ImpostorNormal' );
		const hitDepth = property( 'float', 'ImpostorDepth' );
		const space = this._setupSpace( builder, normal, hitDepth );

		this.colorNode = this.impostor.layout === 'octahedral' ? this._setupOctahedral( space ) : this._setupBox( space );
		this.normalNode = normal;

		super.setupDiffuseColor( builder );

		depth.assign( hitDepth ).toStack();

	}

	setupDepth() {

		// written in setupDiffuseColor(), once the impostor has found the hit

	}

	// The view ray in the object's local space, and how to turn a local hit into a normal and a depth.

	_setupSpace( builder, normal, hitDepth ) {

		const object = builder.object;
		const objectMatrix = object.isInstancedMesh ? modelWorldMatrix.mul( instanceMatrixNode( builder, object.instanceMatrix ) ) : modelWorldMatrix;
		const objectViewMatrix = cameraViewMatrix.mul( objectMatrix );
		const objectMatrixInverse = inverse( objectMatrix );

		// per instance, in the vertex stage: the camera in object space, and the object-to-view transform

		const localPoint = varying( positionGeometry );
		const localEye = varying( objectMatrixInverse.mul( vec4( cameraPosition, 1 ) ).xyz );
		const localForward = varying( objectMatrixInverse.mul( vec4( cameraWorldMatrix.element( 2 ).xyz.negate(), 0 ) ).xyz );
		const viewOrigin = varying( objectViewMatrix.element( 3 ).xyz );
		const viewX = varying( objectViewMatrix.element( 0 ).xyz );
		const viewY = varying( objectViewMatrix.element( 1 ).xyz );
		const viewZ = varying( objectViewMatrix.element( 2 ).xyz );
		const normalMatrix = transpose( inverse( mat3( objectViewMatrix.element( 0 ).xyz, objectViewMatrix.element( 1 ).xyz, objectViewMatrix.element( 2 ).xyz ) ) );
		const normalX = varying( normalMatrix.element( 0 ) );
		const normalY = varying( normalMatrix.element( 1 ) );
		const normalZ = varying( normalMatrix.element( 2 ) );

		const isOrthographic = cameraProjectionMatrix.element( 2 ).w.equal( 0 );

		const { reversedDepthBuffer, logarithmicDepthBuffer } = builder.renderer;
		const perspectiveDepth = reversedDepthBuffer ? viewZToReversedPerspectiveDepth : viewZToPerspectiveDepth;
		const orthographicDepth = reversedDepthBuffer ? viewZToReversedOrthographicDepth : viewZToOrthographicDepth;

		const reach = length( vec3( ...this.impostor.box.getSize( _size ).toArray() ) ).mul( 4 );

		return {

			isOrthographic,
			localEye,
			localForward,

			// ray from the eye (or, for orthographic cameras, from well in front of the box) through this pixel
			ray: () => {

				const origin = select( isOrthographic, localPoint.sub( localForward.mul( reach ) ), localEye ).toVar();
				return { origin, direction: localPoint.sub( origin ).toVar() };

			},

			setNormal: ( localNormal ) => {

				normal.assign( normalize( normalX.mul( localNormal.x ).add( normalY.mul( localNormal.y ) ).add( normalZ.mul( localNormal.z ) ) ) );

			},

			// write the real depth of the hit, so impostors intersect each other and the scene correctly
			setDepth: ( local ) => {

				const hitViewZ = viewOrigin.add( viewX.mul( local.x ) ).add( viewY.mul( local.y ) ).add( viewZ.mul( local.z ) ).z;

				if ( logarithmicDepthBuffer ) {

					hitDepth.assign( viewZToLogarithmicDepth( hitViewZ, cameraNear, cameraFar ) );

				} else {

					hitDepth.assign( select( isOrthographic, orthographicDepth( hitViewZ, cameraNear, cameraFar ), perspectiveDepth( hitViewZ, cameraNear, cameraFar ) ) );

				}

			}

		};

	}

	_setupBox( space ) {

		const { box, resolution, intervalMap, colorMap, normalMap } = this.impostor;
		const boxMin = vec3( ...box.min.toArray() );
		const boxSize = vec3( ...box.getSize( _size ).toArray() );

		const intervalUV = ( b, axis ) => {

			const p = clamp( axisUV[ axis ]( b ), 0.5 / resolution, 1 - 0.5 / resolution );
			return vec2( p.x.add( axis ), p.y ).div( vec2( 3, 1 ) );

		};

		const viewAtlasUV = ( b, i ) => {

			const p = clamp( viewUV[ i ]( b ), 0.5 / resolution, 1 - 0.5 / resolution );
			return vec2( p.x.add( i % 3 ), p.y.add( Math.floor( i / 3 ) ) ).div( vec2( 3, 2 ) );

		};

		const intervalAt = ( b, axis ) => texture( intervalMap, intervalUV( b, axis ) ).level( 0 );

		const epsilon = 1.5 / resolution;
		const inside = ( interval, c ) => c.lessThanEqual( interval.x.add( epsilon ) ).and( c.greaterThanEqual( interval.y.oneMinus().sub( epsilon ) ) );

		// solid = inside the surface interval along all three axes, testing one axis at a time
		const solidAt = ( point ) => {

			const b = point.toVar();
			const solid = bool( false ).toVar();
			const x = intervalAt( b, 0 );

			If( inside( x, b.x ), () => {

				const y = intervalAt( b, 1 );

				If( inside( y, b.y ), () => {

					const z = intervalAt( b, 2 );

					If( inside( z, b.z ), () => {

						solid.assign( true );

					} );

				} );

			} );

			return solid;

		};

		return Fn( () => {

			// the view ray in box space, where the box spans [0, 1] on each axis

			const { origin, direction } = space.ray();
			const o = origin.sub( boxMin ).div( boxSize ).toVar();
			const d = direction.div( boxSize ).toVar();

			const inverseD = vec3( 1 ).div( d );
			const t0 = o.negate().mul( inverseD ).toVar();
			const t1 = o.oneMinus().mul( inverseD ).toVar();
			const tNear = max( max( min( t0.x, t1.x ), min( t0.y, t1.y ) ), max( min( t0.z, t1.z ), 0 ) ).toVar();
			const tFar = min( min( max( t0.x, t1.x ), max( t0.y, t1.y ) ), max( t0.z, t1.z ) ).toVar();

			const hit = bool( false ).toVar();
			const t = float( 0 ).toVar();
			const previous = tNear.toVar();
			const steps = clamp( ceil( length( d ).mul( tFar.sub( tNear ) ).div( 1.2 ).mul( this.stepsNode ) ), 2, this.stepsNode ).toVar();
			const stepSize = tFar.sub( tNear ).div( steps ).toVar();

			If( tFar.greaterThan( tNear ), () => {

				Loop( { start: int( 0 ), end: int( steps ).add( 1 ), type: 'int', condition: '<' }, ( { i } ) => {

					const ti = tNear.add( stepSize.mul( float( i ) ) ).toVar();

					If( solidAt( o.add( d.mul( ti ) ) ), () => {

						hit.assign( true );
						t.assign( ti );
						Break();

					} );

					previous.assign( ti );

				} );

			} );

			If( hit.not(), () => {

				Discard();

			} );

			// refine between the last empty sample and the first solid one

			If( t.greaterThan( tNear ), () => {

				const low = previous.toVar(), high = t.toVar();

				Loop( 6, () => {

					const middle = low.add( high ).mul( 0.5 ).toVar();

					If( solidAt( o.add( d.mul( middle ) ) ), () => {

						high.assign( middle );

					} ).Else( () => {

						low.assign( middle );

					} );

				} );

				t.assign( high );

			} );

			const p = o.add( d.mul( t ) ).toVar();

			// how far p is from the surface each view saw; the closest view is the one that saw p

			const x = intervalAt( p, 0 ), y = intervalAt( p, 1 ), z = intervalAt( p, 2 );
			const distances = [
				abs( x.x.sub( p.x ) ), abs( p.x.sub( x.y.oneMinus() ) ),
				abs( y.x.sub( p.y ) ), abs( p.y.sub( y.y.oneMinus() ) ),
				abs( z.x.sub( p.z ) ), abs( p.z.sub( z.y.oneMinus() ) )
			].map( ( distance ) => distance.toVar() );
			const closest = distances.reduce( ( a, b ) => min( a, b ) ).toVar();

			const color = vec3( 0 ).toVar();
			const localNormal = vec3( 0 ).toVar();
			const weight = float( 1e-5 ).toVar();

			for ( let i = 0; i < 6; i ++ ) {

				const atlasUV = viewAtlasUV( p, i );
				const c = texture( colorMap, atlasUV ).level( 0 );
				const n = texture( normalMap, atlasUV ).level( 0 );
				const w = exp( distances[ i ].sub( closest ).mul( - 0.5 * resolution ) ).mul( c.a ).toVar();

				// texels at silhouette edges are blended with empty space, so undo that
				color.addAssign( c.rgb.div( max( c.a, 1e-3 ) ).mul( w ) );
				localNormal.addAssign( n.rgb.div( max( c.a, 1e-3 ) ).mul( 2 ).sub( 1 ).mul( w ) );
				weight.addAssign( w );

			}

			space.setNormal( localNormal );
			space.setDepth( boxMin.add( p.mul( boxSize ) ) );

			return vec4( color.div( weight ).mul( materialColor ), 1 );

		} )();

	}

	_setupOctahedral( space ) {

		const { box, resolution, frames, hemisphere, radius, colorMap, normalMap } = this.impostor;
		const center = vec3( ...box.getCenter( _center ).toArray() );
		const half = 0.5 / resolution;

		return Fn( () => {

			const { origin, direction } = space.ray();

			// the three views around the direction towards the camera, and their blend weights

			const toCamera = normalize( select( space.isOrthographic, space.localForward.negate(), space.localEye.sub( center ) ) );
			const grid = octahedralEncode( toCamera, hemisphere ).mul( frames - 1 ).toVar();
			const base = clamp( floor( grid ), 0, frames - 2 ).toVar();
			const f = clamp( grid.sub( base ), 0, 1 ).toVar();

			const views = [ base, base.add( 1 ), base.add( select( f.y.greaterThan( f.x ), vec2( 0, 1 ), vec2( 1, 0 ) ) ) ];
			const weights = [ max( f.x, f.y ).oneMinus(), min( f.x, f.y ), abs( f.x.sub( f.y ) ) ];

			const color = vec3( 0 ).toVar();
			const localNormal = vec3( 0 ).toVar();
			const tHit = float( 0 ).toVar();
			const weight = float( 1e-5 ).toVar();
			const coverage = float( 0 ).toVar();

			for ( let k = 0; k < 3; k ++ ) {

				const view = views[ k ].toVar();
				const toView = octahedralDecode( view.div( frames - 1 ), hemisphere ).toVar();
				const up0 = select( abs( toView.y ).greaterThan( 0.999 ), vec3( 0, 0, - 1 ), vec3( 0, 1, 0 ) );
				const right = normalize( cross( up0, toView ) ).toVar();
				const up = cross( toView, right ).toVar();

				const sample = ( point ) => {

					const q = point.sub( center );
					const imageUV = clamp( vec2( dot( q, right ), dot( q, up ) ).div( radius * 2 ).add( 0.5 ), half, 1 - half );
					const atlasUV = vec2( view.x.add( imageUV.x ), view.y.add( imageUV.y.oneMinus() ) ).div( frames );
					return { c: texture( colorMap, atlasUV ).level( 0 ), n: texture( normalMap, atlasUV ).level( 0 ) };

				};

				// meet the plane through the centre facing this view, then step to the stored height

				const facing = dot( direction, toView );
				const along = dot( center.sub( origin ), toView );
				const denominator = select( abs( facing ).lessThan( 1e-6 ), - 1e-6, facing ).toVar();
				const tPlane = along.div( denominator ).toVar();

				const first = sample( origin.add( direction.mul( tPlane ) ) );
				const firstHeight = first.n.a.div( max( first.c.a, 1e-3 ) ).sub( 0.5 ).mul( radius * 2 );
				const tSurface = select( first.c.a.greaterThan( 0.01 ), along.add( firstHeight ).div( denominator ), tPlane ).toVar();

				const s = sample( origin.add( direction.mul( tSurface ) ) );
				const c = s.c.toVar();
				const w = weights[ k ].mul( c.a ).toVar();

				// texels at silhouette edges are blended with empty space, so undo that
				color.addAssign( c.rgb.div( max( c.a, 1e-3 ) ).mul( w ) );
				localNormal.addAssign( s.n.rgb.div( max( c.a, 1e-3 ) ).mul( 2 ).sub( 1 ).mul( w ) );
				tHit.addAssign( tSurface.mul( w ) );
				weight.addAssign( w );
				coverage.addAssign( w );

			}

			If( coverage.lessThan( 0.5 ), () => {

				Discard();

			} );

			space.setNormal( localNormal );
			space.setDepth( origin.add( direction.mul( tHit.div( weight ) ) ) );

			return vec4( color.div( weight ).mul( materialColor ), 1 );

		} )();

	}

	copy( source ) {

		this.impostor = source.impostor;
		this.steps = source.steps;

		return super.copy( source );

	}

}

export { ImpostorBaker, ImpostorData, ImpostorNodeMaterial };
