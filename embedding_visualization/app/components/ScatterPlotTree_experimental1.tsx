'use client';

import React, { useMemo, useRef, useState, Suspense, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html, Stars } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import { useTheme } from 'next-themes';

import type { Point3D, HighlightMap } from '../../lib/types/types';
import { buildCategoryColorMap } from '../../lib/utils/categoryColors';
import { calculateLuminosity, calculateSimilarityColors } from '../../lib/utils/plotUtils';

// --- Constants ---
const GALAXY_RADIUS = 100; // Scale factor for the galaxy view

// --- Types ---
interface ScatterPlot3DProps {
    points: Point3D[];
    colorBy?: 'category' | 'none';
    categoryField?: string | null;
    categoryValues?: string[];
    highlightedIndices?: HighlightMap;
    selectedPoint?: Point3D | null;
    onPointClick?: (point: Point3D) => void;
    className?: string;
    showOnlyHighlighted?: boolean;
    showLabels?: boolean;
}

// --- Components ---

// 1. The Main Galaxy Cloud (All Points)
const GalaxyPoints: React.FC<{
    points: Point3D[];
    colors: Float32Array;
    onPointClick?: (point: Point3D) => void;
    highlightedIndices?: HighlightMap;
    showOnlyHighlighted?: boolean;
}> = ({ points, colors, onPointClick, highlightedIndices, showOnlyHighlighted }) => {
    const meshRef = useRef<THREE.Points>(null!);
    const hoverRef = useRef<number | null>(null);
    const [hoveredPoint, setHoveredPoint] = useState<Point3D | null>(null);

    // Calculate positions once
    const positions = useMemo(() => {
        const pos = new Float32Array(points.length * 3);
        for (let i = 0; i < points.length; i++) {
            // Normalize positions to fit in view if needed, or use as is
            // Assuming incoming points are somewhat normalized or we scale camera
            pos[i * 3] = points[i].x;
            pos[i * 3 + 1] = points[i].y;
            pos[i * 3 + 2] = points[i].z;
        }
        return pos;
    }, [points]);

    // Handle visibility based on showOnlyHighlighted
    // We can't easily hide individual points in a single draw call without updating geometry or using a shader
    // Simplest approach: Move hidden points to infinity or set alpha to 0 (requires custom shader or attr update)
    // For now, if showOnlyHighlighted is true, we simply don't render this layer if we want to hide everything else
    // BUT the user might want to see context. 
    // Let's rely on the opacity trick: if showOnlyHighlighted, we lower opacity of non-highlights drastically.

    // Actually, standard PointsMaterial doesn't support per-vertex opacity easily without custom shaders.
    // We'll stick to rendering everything for context, maybe lighter.

    // Interaction handlers
    const handlePointerMove = (e: any) => {
        e.stopPropagation();
        const index = e.index;
        if (index !== undefined && index !== hoverRef.current) {
            hoverRef.current = index;
            setHoveredPoint(points[index]);
            document.body.style.cursor = 'pointer';
        }
    };

    const handlePointerOut = () => {
        hoverRef.current = null;
        setHoveredPoint(null);
        document.body.style.cursor = 'auto';
    };

    const handleClick = (e: any) => {
        e.stopPropagation();
        if (onPointClick && e.index !== undefined) {
            onPointClick(points[e.index]);
        }
    };

    return (
        <group>
            <points
                ref={meshRef}
                onPointerMove={handlePointerMove}
                onPointerOut={handlePointerOut}
                onClick={handleClick}
            >
                <bufferGeometry>
                    <bufferAttribute
                        attach="attributes-position"
                        count={positions.length / 3}
                        array={positions}
                        itemSize={3}
                    />
                    <bufferAttribute
                        attach="attributes-color"
                        count={colors.length / 3}
                        array={colors}
                        itemSize={3}
                    />
                </bufferGeometry>
                <pointsMaterial
                    size={showOnlyHighlighted ? 0.05 : 0.15} // Shrink background if focused
                    sizeAttenuation
                    vertexColors
                    transparent
                    opacity={showOnlyHighlighted ? 0.1 : 0.8}
                    depthWrite={false}
                />
            </points>

            {/* Hover Tooltip Overlay */}
            {hoveredPoint && (
                <Html position={[hoveredPoint.x, hoveredPoint.y, hoveredPoint.z]} zIndexRange={[100, 0]}>
                    <div className="pointer-events-none px-2 py-1 bg-black/80 backdrop-blur-md text-white text-xs rounded border border-white/10 whitespace-nowrap transform -translate-y-full -translate-x-1/2 -mt-2">
                        <div className="font-bold">{hoveredPoint.label}</div>
                        <div className="opacity-70 max-w-[200px] truncate">{hoveredPoint.document}</div>
                    </div>
                </Html>
            )}
        </group>
    );
};

// 2. Highlighted Points (Glow Layer)
const HighlightedPoints: React.FC<{
    points: Point3D[];
    highlightedIndices: HighlightMap;
    selectedPoint?: Point3D | null;
    onPointClick?: (point: Point3D) => void;
}> = ({ points, highlightedIndices, selectedPoint, onPointClick }) => {

    // Filter only relevant points
    const highlightData = useMemo(() => {
        if (!highlightedIndices || highlightedIndices.size === 0) return [];

        return points
            .filter(p => highlightedIndices.has(p.index))
            .map(p => {
                const similarity = highlightedIndices.get(p.index) || 0;
                const colors = calculateSimilarityColors(similarity);
                return {
                    point: p,
                    color: new THREE.Color(colors.glowColor),
                    similarity
                };
            });
    }, [points, highlightedIndices]);

    if (highlightData.length === 0) return null;

    return (
        <group>
            {highlightData.map(({ point, color, similarity }, i) => (
                <mesh
                    key={point.id}
                    position={[point.x, point.y, point.z]}
                    onClick={(e) => {
                        e.stopPropagation();
                        onPointClick?.(point);
                    }}
                >
                    <sphereGeometry args={[0.3, 16, 16]} />
                    <meshBasicMaterial color={color} />
                    {/* Add a glow sprite or bigger mesh here if preferred */}

                    {/* Label */}
                    <Html distanceFactor={10} zIndexRange={[50, 0]}>
                        <div className="text-[10px] text-white/90 font-mono bg-black/40 px-1 rounded pointer-events-none whitespace-nowrap transform -translate-x-1/2 -translate-y-[150%]">
                            {point.label}
                        </div>
                    </Html>
                </mesh>
            ))}
        </group>
    );
};

// 3. Camera Controller (Fly to selected)
const CameraController: React.FC<{ selectedPoint?: Point3D | null }> = ({ selectedPoint }) => {
    const { camera, controls } = useThree();
    const targetRef = useRef(new THREE.Vector3(0, 0, 0));

    useFrame((state, delta) => {
        if (selectedPoint) {
            const targetPos = new THREE.Vector3(selectedPoint.x, selectedPoint.y, selectedPoint.z);

            // If the target has changed significantly, start moving
            if (targetPos.distanceTo(targetRef.current) > 0.1) {
                targetRef.current.copy(targetPos);
            }

            const controlsImpl = (controls as any)?.current;
            if (controlsImpl) {
                // Smoothly move controls target to the point
                controlsImpl.target.lerp(targetPos, 4 * delta);

                // Optional: Do we move the camera too? 
                // Usually we want to look AT the point from a distance, not be ON the point.
                // Let's just keep the camera orbiting existing distance but re-center focus.
                controlsImpl.update();
            }
        }
    });

    return null;
};


// 4. Main Scene Wrapper
const Scene: React.FC<ScatterPlot3DProps> = ({
    points,
    colorBy,
    categoryField,
    categoryValues,
    highlightedIndices,
    selectedPoint,
    onPointClick,
    showOnlyHighlighted,
    showLabels
}) => {
    const { resolvedTheme } = useTheme();
    const colorMap = useMemo(() => buildCategoryColorMap(categoryField, categoryValues), [categoryField, categoryValues]);

    // Prepare Colors Buffer
    const colors = useMemo(() => {
        const cols = new Float32Array(points.length * 3);
        const defaultColor = new THREE.Color("#1f77b4");

        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            let c = defaultColor;

            if (colorBy === 'category' && p.category) {
                const hex = colorMap[p.category];
                if (hex) c = new THREE.Color(hex);
            }

            cols[i * 3] = c.r;
            cols[i * 3 + 1] = c.g;
            cols[i * 3 + 2] = c.b;
        }
        return cols;
    }, [points, colorBy, colorMap]);

    return (
        <>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1} />

            <Stars radius={300} depth={100} count={5000} factor={4} saturation={0} fade speed={1} />

            <GalaxyPoints
                points={points}
                colors={colors}
                onPointClick={onPointClick}
                highlightedIndices={highlightedIndices}
                showOnlyHighlighted={showOnlyHighlighted}
            />

            {highlightedIndices && highlightedIndices.size > 0 && (
                <HighlightedPoints
                    points={points}
                    highlightedIndices={highlightedIndices}
                    selectedPoint={selectedPoint}
                    onPointClick={onPointClick}
                />
            )}

            {selectedPoint && (
                // Special marker for the actively selected point
                <mesh position={[selectedPoint.x, selectedPoint.y, selectedPoint.z]}>
                    <sphereGeometry args={[0.5, 32, 32]} />
                    <meshStandardMaterial color="#ffcc00" emissive="#ffaa00" emissiveIntensity={2} />
                    <Html>
                        <div className="w-4 h-4 border-2 border-yellow-400 rounded-full animate-ping absolute -top-2 -left-2" />
                    </Html>
                </mesh>
            )}

            <CameraController selectedPoint={selectedPoint} />
            <OrbitControls makeDefault enableDamping dampingFactor={0.1} />

            <EffectComposer>
                <Bloom luminanceThreshold={1} result={undefined} intensity={1.5} />
            </EffectComposer>
        </>
    );
};

export function ScatterPlot3D(props: ScatterPlot3DProps) {
    return (
        <div className={`w-full h-full bg-[#08080b] ${props.className || ''}`}>
            <Canvas camera={{ position: [0, 0, 50], fov: 60 }}>
                <Suspense fallback={<Html center>Loading Galaxy...</Html>}>
                    <Scene {...props} />
                </Suspense>
            </Canvas>
        </div>
    );
}