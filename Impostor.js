import {
	BackSide,
	Box3,
	BoxGeometry,
	Color,
	DoubleSide,
	DynamicDrawUsage,
	HalfFloatType,
	InstancedInterleavedBuffer,
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
	depth,
	diffuseColor,
	exp,
	float,
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

// The six views, each looking into the box from one side. `width` and `height` are the box axes
// that map to the image's horizontal and vertical directions.
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

const _center = /*@__PURE__*/ new Vector3();
const _size = /*@__PURE__*/ new Vector3();
const _clearColor = /*@__PURE__*/ new Color();

/**
 * The result of {@link ImpostorBaker#bake}: the textures that describe an object, plus the box they
 * were captured in. Pass it to {@link ImpostorNodeMaterial} and draw it with {@link ImpostorData#geometry}.
 */
class ImpostorData {

	constructor( box, resolution, colorTarget, normalTarget, intervalTarget ) {

		this.isImpostorData = true;

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
		 * Base colour of the six views, in a 3 x 2 grid.
		 *
		 * @type {Texture}
		 */
		this.colorMap = colorTarget.texture;

		/**
		 * Object-space normals of the six views, in a 3 x 2 grid.
		 *
		 * @type {Texture}
		 */
		this.normalMap = normalTarget.texture;

		/**
		 * For each axis, the nearest surface seen from the positive side (red) and from the
		 * negative side (green, stored as one minus the coordinate), in a 3 x 1 grid.
		 *
		 * @type {Texture}
		 */
		this.intervalMap = intervalTarget.texture;

		/**
		 * A box matching {@link ImpostorData#box}, to render the impostor with.
		 *
		 * @type {BoxGeometry}
		 */
		this.geometry = new BoxGeometry( ...box.getSize( _size ).toArray() ).translate( ...box.getCenter( _center ).toArray() );

		this._targets = [ colorTarget, normalTarget, intervalTarget ];

	}

	dispose() {

		for ( const target of this._targets ) target.dispose();
		this.geometry.dispose();

	}

}

/**
 * Captures an object from six sides so it can be drawn as an impostor with {@link ImpostorNodeMaterial}.
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
	 * @param {Object3D} object - The object to capture.
	 * @param {Object} [options]
	 * @param {number} [options.resolution=256] - The size in pixels of each view.
	 * @return {Promise<ImpostorData>}
	 */
	async bake( object, { resolution = 256 } = {} ) {

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
		const size = box.getSize( new Vector3() );
		const extent = Math.max( size.x, size.y, size.z );
		box.expandByScalar( extent * 0.02 );
		box.getSize( size );
		const center = box.getCenter( new Vector3() );

		// capture the six views: base colour, normal, and the surface position in box space

		const boxPosition = positionWorld.sub( vec3( ...box.min.toArray() ) ).div( vec3( ...size.toArray() ) );
		const worldNormal = normalize( cameraWorldMatrix.mul( vec4( normalView, 0 ) ).xyz );

		const views = VIEWS.map( () => {

			const target = new RenderTarget( resolution, resolution, { count: 3, type: HalfFloatType } );
			target.textures[ 0 ].name = 'output';
			target.textures[ 1 ].name = 'normal';
			target.textures[ 2 ].name = 'position';
			return target;

		} );

		const previousTarget = renderer.getRenderTarget();
		const previousMRT = renderer.getMRT();
		const previousAutoClear = renderer.autoClear;
		const previousClearAlpha = renderer.getClearAlpha();
		renderer.getClearColor( _clearColor );

		renderer.autoClear = true;
		renderer.setClearColor( 0x000000, 0 );
		renderer.setMRT( mrt( {
			output: vec4( diffuseColor.rgb, 1 ),
			normal: vec4( worldNormal.mul( 0.5 ).add( 0.5 ), 1 ),
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
			camera.position.copy( center ).addScaledVector( new Vector3( ...view.side ), half + margin );
			camera.up.set( ...view.up );
			camera.lookAt( center );
			camera.updateProjectionMatrix();
			camera.updateMatrixWorld();

			renderer.setRenderTarget( views[ i ] );
			renderer.render( scene, camera );

		} );

		renderer.setMRT( null );

		// pack the views into the colour and normal atlases, and opposite views into the interval atlas

		const quad = new PlaneGeometry();
		const atlas = ( width, height ) => new RenderTarget( width * resolution, height * resolution, { type: HalfFloatType, depthBuffer: false } );
		const tile = ( column, row, columns, rows ) => {

			// render targets are sampled top row first, so tiles are written upside down
			const p = vec2( column, row ).add( uv() ).div( vec2( columns, rows ) ).mul( 2 ).sub( 1 );
			return vec4( p.x, p.y.negate(), 0, 1 );

		};

		const pack = ( target, quads ) => {

			const packScene = new Scene();

			for ( const { vertex, fragment } of quads ) {

				const material = new NodeMaterial();
				material.side = DoubleSide; // the flipped tile reverses the winding
				material.depthTest = false;
				material.depthWrite = false;
				material.vertexNode = vertex;
				material.fragmentNode = fragment;

				const mesh = new Mesh( quad, material );
				mesh.frustumCulled = false;
				packScene.add( mesh );

			}

			renderer.setRenderTarget( target );
			renderer.render( packScene, camera );

			packScene.traverse( ( child ) => child.material?.dispose() );

		};

		// render targets are sampled top row first, so flip v to read a capture in image space (v up)
		const capture = ( map, imageUV ) => texture( map, vec2( imageUV.x, imageUV.y.oneMinus() ) );

		const colorTarget = atlas( 3, 2 );
		const normalTarget = atlas( 3, 2 );
		const intervalTarget = atlas( 3, 1 );

		pack( colorTarget, views.map( ( view, i ) => ( { vertex: tile( i % 3, Math.floor( i / 3 ), 3, 2 ), fragment: capture( view.textures[ 0 ], uv() ) } ) ) );
		pack( normalTarget, views.map( ( view, i ) => ( { vertex: tile( i % 3, Math.floor( i / 3 ), 3, 2 ), fragment: capture( view.textures[ 1 ], uv() ).mul( capture( view.textures[ 0 ], uv() ).a ) } ) ) );

		const st = uv();
		const flipU = vec2( st.x.oneMinus(), st.y );
		const flipV = vec2( st.x, st.y.oneMinus() );
		const positiveUV = [ flipU, flipV, st ];
		const negativeUV = [ st, st, flipU ];

		pack( intervalTarget, [ 0, 1, 2 ].map( ( axis ) => {

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

		renderer.setRenderTarget( previousTarget );
		renderer.setMRT( previousMRT );
		renderer.autoClear = previousAutoClear;
		renderer.setClearColor( _clearColor, previousClearAlpha );

		for ( const view of views ) view.dispose();
		quad.dispose();

		return new ImpostorData( box, resolution, colorTarget, normalTarget, intervalTarget );

	}

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
 * Each pixel marches a ray through the captured box. A point is solid when it lies between the
 * nearest surfaces seen from both sides of every axis, which costs up to three texture reads.
 * The hit is shaded with the captured colour and normal through the usual three.js lighting,
 * and writes its real depth.
 *
 * Shadows are not supported yet.
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
		 * Uniform holding the largest number of ray steps for a pixel.
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
	 * The largest number of ray steps for a pixel.
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

		// the march runs as part of the base colour, which three.js builds before the normal is needed;
		// it also produces the normal and the depth of the hit
		const normal = property( 'vec3', 'ImpostorNormal' );
		const hitDepth = property( 'float', 'ImpostorDepth' );

		this.colorNode = this._setupMarch( builder, normal, hitDepth );
		this.normalNode = normal;

		super.setupDiffuseColor( builder );

		depth.assign( hitDepth ).toStack();

	}

	setupDepth() {

		// written in setupDiffuseColor(), once the march has found the hit

	}

	_setupMarch( builder, normal, hitDepth ) {

		const { box, resolution } = this.impostor;
		const boxMin = vec3( ...box.min.toArray() );
		const boxSize = vec3( ...box.getSize( _size ).toArray() );

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

		const intervalUV = ( b, axis ) => {

			const p = clamp( axisUV[ axis ]( b ), 0.5 / resolution, 1 - 0.5 / resolution );
			return vec2( p.x.add( axis ), p.y ).div( vec2( 3, 1 ) );

		};

		const viewAtlasUV = ( b, i ) => {

			const p = clamp( viewUV[ i ]( b ), 0.5 / resolution, 1 - 0.5 / resolution );
			return vec2( p.x.add( i % 3 ), p.y.add( Math.floor( i / 3 ) ) ).div( vec2( 3, 2 ) );

		};

		const intervalAt = ( b, axis ) => texture( this.impostor.intervalMap, intervalUV( b, axis ) ).level( 0 );

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

			const origin = select( isOrthographic, localPoint.sub( localForward.mul( length( boxSize ).mul( 4 ) ) ), localEye );
			const o = origin.sub( boxMin ).div( boxSize ).toVar();
			const d = localPoint.sub( origin ).div( boxSize ).toVar();

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
				const c = texture( this.impostor.colorMap, atlasUV ).level( 0 );
				const n = texture( this.impostor.normalMap, atlasUV ).level( 0 );
				const w = exp( distances[ i ].sub( closest ).mul( - 0.5 * resolution ) ).mul( c.a ).toVar();

				// texels at silhouette edges are blended with empty space, so undo that
				color.addAssign( c.rgb.div( max( c.a, 1e-3 ) ).mul( w ) );
				localNormal.addAssign( n.rgb.div( max( c.a, 1e-3 ) ).mul( 2 ).sub( 1 ).mul( w ) );
				weight.addAssign( w );

			}

			normal.assign( normalize( normalX.mul( localNormal.x ).add( normalY.mul( localNormal.y ) ).add( normalZ.mul( localNormal.z ) ) ) );

			// write the real depth of the hit, so impostors intersect each other and the scene correctly

			const local = boxMin.add( p.mul( boxSize ) );
			const hitViewZ = viewOrigin.add( viewX.mul( local.x ) ).add( viewY.mul( local.y ) ).add( viewZ.mul( local.z ) ).z;

			if ( logarithmicDepthBuffer ) {

				hitDepth.assign( viewZToLogarithmicDepth( hitViewZ, cameraNear, cameraFar ) );

			} else {

				hitDepth.assign( select( isOrthographic, orthographicDepth( hitViewZ, cameraNear, cameraFar ), perspectiveDepth( hitViewZ, cameraNear, cameraFar ) ) );

			}

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
