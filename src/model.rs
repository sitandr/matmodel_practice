use egui::{debug_text::print, vec2, Color32, ColorImage, Image, TextureOptions, Vec2};
use itertools::Itertools;

type Coord = (usize, usize);

enum Neighbor {
    Left,
    Right,
    Upper,
    Bottom,
}

impl Neighbor {
    #[inline]
    fn get_v(&self, parent: &Modelling, i: usize) -> f64 {
        parent.data[self.get_i(&parent.geometry, i)]
    }

    #[inline]
    fn get_i(&self, geometry: &Geometry, i: usize) -> usize {
        match self {
            Neighbor::Left => i - 1,
            Neighbor::Right => i + 1,
            Neighbor::Upper => i - geometry.width,
            Neighbor::Bottom => i + geometry.width
        }
    }
}

pub enum EquationType {
    Wave,
    Heat
}



#[derive(Clone, Copy)]
struct Geometry {
    width: usize,
    height: usize,
}

pub struct FluxSetup {
    pub flux: f64,
    pub alpha: f64,
    pub beta: f64,
    pub left: f64,
    pub right: f64,
    pub up: f64,
    pub down: f64,
}

impl FluxSetup {
    fn get_flux(&self, dir: &Neighbor) -> f64 {
        self.flux * match dir {
            Neighbor::Left => self.right,
            Neighbor::Right => self.left,
            Neighbor::Upper => self.down,
            Neighbor::Bottom => self.up,
        }
    }
}

impl Geometry {
    /// returns direction normal to edge "into" the data 
    fn edge_type(&self, i: usize) -> Option<Neighbor> {
        let (x, y) = self.index2coord(i);
        let last_x = self.width - 1;
        let last_y = self.height - 1;
        match (x, y) {
            (0, _) => Some(Neighbor::Right),
            (last, _) if last == last_x => Some(Neighbor::Left),
            (_, 0) => Some(Neighbor::Bottom),
            (_, last) if last == last_y => Some(Neighbor::Upper),
            _ => None,
        }
    }

    fn is_angle(&self, i: usize) -> bool {
        let (x, y) = self.index2coord(i);
        let last_x = self.width - 1;
        let last_y = self.height - 1;

        let cond1 = ((last_x - x) <= 1) as u8;
        let cond2 = ((last_y - y) <= 1) as u8;
        let cond3 = (x <= 1) as u8;
        let cond4 = (y <= 1) as u8;

        cond1 + cond2 + cond3 + cond4 >= 2
    }
    
    #[inline]
    fn index2coord(&self, i: usize) -> Coord {
        (i % self.width, i / self.width)
    }

    fn ind_iter(&self) -> impl Iterator<Item=usize> {
        0..(self.width*self.height)
    }

    fn coord2index(&self, (x, y): (i32, i32)) -> usize {
        // if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
        // return None;
        // }
        x as usize + y as usize * self.width
    }
}

pub struct Modelling {
    geometry: Geometry,
    /// <-> width, height ↑↓
    pub data: Vec<f64>,
    prev_data: Vec<f64>,
    texture: Option<egui::TextureHandle>,
    buffer: Vec<f64>,
    pub c0: f64,
    pub eq_type: EquationType,
    pub flux: FluxSetup,
    pub automax: bool,
    pub max: f64,
    pub min: f64
}

impl Modelling {
    pub fn init(width: usize, height: usize) -> Self {
        let mut data = vec![0.0; width * height];
        let buffer = Vec::with_capacity(width * height);
        data[520] = 1.0;
        let prev_data = data.clone();

        Self {
            geometry: Geometry { width, height },
            data,
            prev_data,
            buffer,
            c0: 1.0,
            max: 1.0,
            min: -1.0,
            eq_type: EquationType::Heat,
            flux: FluxSetup { flux: 0.5, alpha: 0.5, beta: 0.0, left: 1.0, right: -1.0, up: 0.0, down: 0.0},
            texture: None
        }
    }

    pub fn step(&mut self, n: usize, dx: f64, dt: f64) {
        for _ in 0..n {
            self.buffer.clear();
            for (i, u) in self.data.iter().enumerate() {
                // ∂_t u = ∂²_x u + ∂²_y u
                // (u_n - u_0)/dt = (u_1 - 2u_0 + u_-1)/dx²
                let is_edge = self.geometry.edge_type(i).is_some();
                let res = if is_edge {
                    *u
                } else {
                    *u + dt * self.c0 * self.laplace(i, *u, dx)
                };
                self.buffer.push(res);
            }

            std::mem::swap(&mut self.data, &mut self.prev_data);
            std::mem::swap(&mut self.buffer, &mut self.data);
            for (i, t) in self.geometry.ind_iter().filter_map(|i| self.geometry.edge_type(i).map(|t| (i, t))) {
                let flux = self.flux.get_flux(&t);
                if self.flux.alpha == 0.0 {
                    self.data[i] = flux;
                } else {
                    /*if self.geometry.is_angle(i) {
                        continue;
                    }*/
                    let prev = t.get_v(&self, t.get_i(&self.geometry, i));
                    // let av = self.data[i] + t.get_v(self, i) + self.data[prev];
                    let delta = (flux - self.flux.beta * prev) / self.flux.alpha * dx;
                    // println!("{av} {delta}");
                    // panic!();
                    self.data[i] = prev + 2.0*delta;
                    // self.data[t.get_i(&self.geometry, i)] = av;
                    // self.data[prev] = av - delta;
                    
                    // self.data[t.get_i(&self.geometry, t.get_i(&self.geometry, i))] = av - delta;
                }
            }
        }
    }

    #[inline]
    fn laplace(&self, i: usize, u: f64, dx: f64) -> f64 {
        (Neighbor::Left.get_v(self, i) - 2.0*u + Neighbor::Right.get_v(self, i)
            + Neighbor::Upper.get_v(self, i) - 2.0*u + Neighbor::Bottom.get_v(self, i)) / (dx * dx)
    }

    pub fn display(&mut self, ui: &mut egui::Ui) {
        let (min, max);
        if self.automax {
            (min, max) = self.data.iter().copied().minmax_by(f64::total_cmp).into_option().unwrap();
        } 
        else{
            max = self.max;//if max < 1.0 {1.0} else {max};
            min = self.min;//if min > 0.0 {0.0} else {min};
        }
        let texture: &mut egui::TextureHandle = self.texture.get_or_insert_with(|| {
            // Load the texture only once.
            ui.ctx().load_texture(
                "my-image",
                egui::ColorImage::example(),
                Default::default()
            )
        });
        let text: Vec<Color32> = self.data.iter().map(|v: &f64| Color32::from_gray(((v.clamp(min, max) - min) / (max - min + 0.001) * 255.0) as u8)).collect();
        texture.set(ColorImage{pixels: text, size: [self.geometry.width, self.geometry.height] }, TextureOptions::NEAREST);

        // Show the image:
        ui.add(Image::new((texture.id(), Vec2::new(500.0, 500.0))));

    }

    /*#[inline]
    /// flux should be u' 
    fn neyman_edge(&self, ind: usize, flux: f64, dx: f64, dt: f64) -> Option<f64> {
        let u = self.data[ind];
        todo!();
        let (x, y) = self.index2coord(ind);
        let last_x = self.width - 1;
        let last_y = self.height - 1;
        let n = match (x, y) {
            (0, _) => Some((Neighbor::Right.get_v(self, ind), flux)),
            (last, _) if last == last_x => Some((Neighbor::Left.get_v(self, ind), -flux)),
            (_, 0) => Some((Neighbor::Bottom.get_v(self, ind), flux)),
            (_, last) if last == last_y => Some((Neighbor::Upper.get_v(self, ind), flux)),
            _ => None,
        };
        n.map(|(n, flux)| {
            let flow = ((n - u)/dx + flux) * dt;
            u + flow
            //0.0
        })
    }*/
}
