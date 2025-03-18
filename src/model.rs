use egui::{Color32, ColorImage, Image, TextureOptions, Vec2};
use evalexpr::{
    build_operator_tree, error::EvalexprResultValue, Context, DefaultNumericTypes, EvalexprError, EvalexprResult, Node, Value
};
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
            Neighbor::Bottom => i + geometry.width,
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Default, PartialEq, Debug)]
pub enum EquationType {
    #[default]
    Wave,
    Heat,
    WaveInnerParam,
    HeatInnerParam
}

#[derive(serde::Deserialize, serde::Serialize, Default, Clone, Copy)]
struct Geometry {
    width: usize,
    height: usize,
}

#[derive(serde::Deserialize, serde::Serialize, Default)]
#[serde(default)]
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
        self.flux
            * match dir {
                Neighbor::Left => self.right,
                Neighbor::Right => self.left,
                Neighbor::Upper => self.down,
                Neighbor::Bottom => self.up,
            }
    }
}

struct FastContext {
    g: Geometry,
    dx: Value,
    xval: Value,
    yval: Value,
    cx: Value,
    cy: Value,
    nx: Value,
    ny: Value,
    w: Value,
    h: Value,
    nw: Value,
    nh: Value,
}

impl FastContext {
    fn by_index(i: usize, g: Geometry, dx: f64) -> Self {
        let (x, y) = g.index2coord(i);

        Self {
            g,
            dx: Value::Float(dx),
            cx: Value::Float(((x as f64 + 0.5) - g.width as f64 / 2.0) * dx),
            cy: Value::Float(((y as f64 + 0.5) - g.height as f64 / 2.0) * dx),
            nx: Value::Int(x as i64),
            ny: Value::Int(y as i64),
            xval: Value::Float((x as f64 + 0.5) * dx),
            yval: Value::Float((y as f64 + 0.5) * dx),
            w: Value::Float(g.width as f64 * dx),
            h: Value::Float(g.height as f64 * dx),
            nw: Value::Int(g.width as i64),
            nh: Value::Int(g.height as i64),
        }
    }

    fn change_i(&mut self, i: usize, dx: f64) {
        let (x, y) = self.g.index2coord(i);
        self.cx = Value::Float(((x as f64 + 0.5) - self.g.width as f64 / 2.0) * dx);
        self.cy = Value::Float(((y as f64 + 0.5) - self.g.height as f64 / 2.0) * dx);
        self.nx = Value::Int(x as i64);
        self.ny = Value::Int(y as i64);
        self.xval = Value::Float((x as f64 + 0.5) * dx);
        self.yval = Value::Float((y as f64 + 0.5) * dx);
    }
}

struct FContext {
    f: Value
}

impl FContext {
    fn new(f: f64) -> Self {
        Self{f: Value::Float(f)}
    }
}


impl Context for FastContext {
    type NumericTypes = DefaultNumericTypes;

    #[inline]
    fn get_value(&self, id: &str) -> Option<&Value<Self::NumericTypes>> {
        match id {
            "x" => Some(&self.xval),
            "y" => Some(&self.yval),
            "cx" => Some(&self.cx),
            "cy" => Some(&self.cy),
            "nx" => Some(&self.nx),
            "ny" => Some(&self.ny),
            "dx" => Some(&self.dx),
            "w" => Some(&self.w),
            "h" => Some(&self.h),
            "nw" => Some(&self.nw),
            "nh" => Some(&self.nh),
            _ => None,
        }
    }

    fn call_function(
        &self,
        identifier: &str,
        _argument: &Value<Self::NumericTypes>,
    ) -> EvalexprResultValue<Self::NumericTypes> {
        Err(EvalexprError::FunctionIdentifierNotFound(
            identifier.to_string(),
        ))
    }

    /// Builtin functions are always enabled
    fn are_builtin_functions_disabled(&self) -> bool {
        false
    }

    /// Builtin functions can't be disabled
    fn set_builtin_functions_disabled(
        &mut self,
        _: bool,
    ) -> EvalexprResult<(), Self::NumericTypes> {unimplemented!()}
}

impl Context for FContext {
    type NumericTypes = DefaultNumericTypes;

    #[inline]
    fn get_value(&self, id: &str) -> Option<&Value<Self::NumericTypes>> {
        match id {
            "f" => Some(&self.f),
            _ => None,
        }
    }

    fn call_function(
        &self,
        identifier: &str,
        _argument: &Value<Self::NumericTypes>,
    ) -> EvalexprResultValue<Self::NumericTypes> {
        Err(EvalexprError::FunctionIdentifierNotFound(
            identifier.to_string(),
        ))
    }

    fn are_builtin_functions_disabled(&self) -> bool {
        false
    }

    fn set_builtin_functions_disabled(
        &mut self,
        _: bool,
    ) -> EvalexprResult<(), Self::NumericTypes> {unimplemented!()}

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

    #[allow(dead_code)]
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

    fn ind_iter(&self) -> impl Iterator<Item = usize> {
        0..(self.width * self.height)
    }

    fn fill_by_expr(&self, data: &mut Vec<f64>, expr: &str, dx: f64) -> Result<(), EvalexprError> {
        let expr = build_operator_tree::<DefaultNumericTypes>(&expr)?;

        let mut context = FastContext::by_index(0, self.clone(), dx);
        let _: Option<()> = data
            .iter_mut()
            .enumerate()
            .filter_map(|(i, v)| {
                context.change_i(i, dx);
                let r = expr.eval_number_with_context(&context);
                match r {
                    Ok(r) => {
                        *v = r;
                        None
                    }
                    Err(e) => return Some(e),
                }
            })
            .next()
            .map(|v| Err(v))
            .transpose()?;

        Ok(())
    }

    #[allow(dead_code)]
    fn coord2index(&self, (x, y): (i32, i32)) -> usize {
        // if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
        // return None;
        // }
        x as usize + y as usize * self.width
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct Modelling {
    geometry: Geometry,
    /// <-> width, height ↑↓
    pub data: Vec<f64>,
    prev_data: Vec<f64>,
    param_x: Vec<f64>,
    #[serde(skip)]
    param_f: Option<Node>,
    #[serde(skip)]
    texture: Option<egui::TextureHandle>,
    buffer: Vec<f64>,
    pub c0: f64,
    pub eq_type: EquationType,
    pub flux: FluxSetup,
    pub automax: bool,
    pub max: f64,
    pub min: f64,
    pub dt: f64,
    pub dx: f64,
}

impl Default for Modelling {
    fn default() -> Self {
        Self {
            geometry: Geometry {
                width: 0,
                height: 0,
            },
            data: Vec::new(),
            prev_data: Vec::new(),
            buffer: Vec::new(),
            param_x: Vec::new(),
            param_f: None,
            c0: 1.0,
            automax: false,
            max: 1.0,
            min: -1.0,
            eq_type: EquationType::Heat,
            flux: FluxSetup {
                flux: 0.5,
                alpha: 0.5,
                beta: 0.0,
                left: 1.0,
                right: -1.0,
                up: 0.0,
                down: 0.0,
            },
            texture: None,
            dt: 0.01,
            dx: 0.5,
        }
    }
}

impl Modelling {
    pub fn init(width: usize, height: usize) -> Self {
        let data = vec![0.0; width * height];
        let buffer = Vec::with_capacity(width * height);
        let prev_data = data.clone();
        let param_x = vec![1.0; width * height];

        Self {
            geometry: Geometry { width, height },
            data,
            prev_data,
            buffer,
            param_x,
            ..Default::default()
        }
    }

    pub fn reset(&mut self, width: usize, height: usize) {
        self.geometry = Geometry{width, height};
        self.data = vec![0.0; self.geometry.width * self.geometry.height];
        self.prev_data = self.data.clone();
        self.param_x = vec![1.0; self.geometry.width * self.geometry.height]
    }

    pub fn step(&mut self, n: usize, dx: f64, dt: f64) -> EvalexprResult<()> {
        for _ in 0..n {
            self.buffer.clear();
            for (i, u) in self.data.iter().enumerate() {
                // ∂_t u = ∂²_x u + ∂²_y u
                // (u_n - u_0)/dt = (u_1 - 2u_0 + u_-1)/dx²
                let is_edge = self.geometry.edge_type(i).is_some();
                let res = if is_edge {
                    *u
                } else {
                    let mut v = self.c0 * self.param_x[i] * self.eval_param_f(*u)? * self.laplace(i, *u, dx);
                    match self.eq_type {
                        EquationType::WaveInnerParam | EquationType::HeatInnerParam => {
                            let dg = self.grad(&self.data, i, dx);
                            v += dg.scal(self.grad(&self.param_x, i, dx)) + dg.square() * self.eval_param_f_der(*u)?;
                        },
                        _ => {}
                    }
                    match self.eq_type {
                        EquationType::Heat | EquationType::HeatInnerParam => *u + dt * v,
                        EquationType::Wave | EquationType::WaveInnerParam => *u * 2.0 + v * dt * dt - self.prev_data[i],
                    }
                };
                self.buffer.push(res);
            }

            std::mem::swap(&mut self.data, &mut self.prev_data);
            std::mem::swap(&mut self.buffer, &mut self.data);

            for (i, t) in self
                .geometry
                .ind_iter()
                .filter_map(|i| self.geometry.edge_type(i).map(|t| (i, t)))
            {
                let flux = self.flux.get_flux(&t);
                if self.flux.alpha == 0.0 {
                    self.data[i] = flux;
                } else {
                    if flux == 0.0 {
                        continue
                    }
                    let prev = t.get_v(&self, t.get_i(&self.geometry, i));
                    let delta = (flux - self.flux.beta * prev) / self.flux.alpha * dx;
                    self.data[i] = prev + 2.0 * delta;
                    // self.data[t.get_i(&self.geometry, t.get_i(&self.geometry, i))] = av - delta;
                }
            }
        }
        Ok(())
    }

    fn eval_param_f_der(&self, f: f64) -> EvalexprResult<f64>{
        if self.param_f.is_some() {
            Ok((self.eval_param_f(f + 0.001)? - self.eval_param_f(f)?) / 0.001)
        } else {Ok(0.0)}
    }

    #[inline]
    fn eval_param_f(&self, f: f64) -> EvalexprResult<f64> {
        self.param_f.as_ref().map(|expr| expr.eval_number_with_context(&FContext::new(f))).unwrap_or(Ok(1.0))
    }

    pub fn set_f(&mut self, expr: &str, dx: f64) -> Result<(), EvalexprError> {
        self.geometry.fill_by_expr(&mut self.data, expr, dx)?;
        self.prev_data = self.data.clone();
        Ok(())
    }

    pub fn set_param(&mut self, expr: &str, dx: f64) -> Result<(), EvalexprError> {
        self.geometry.fill_by_expr(&mut self.param_x, expr, dx)?;
        Ok(())
    }

    pub fn set_param_f(&mut self, expr: &str, enable: bool) -> EvalexprResult<()> {
        self.param_f = enable.then(|| build_operator_tree(expr)).transpose()?;
        Ok(())
    }

    #[inline]
    fn grad(&self, data: &Vec<f64>, i: usize, dx: f64) -> Vect {
        Vect {
            x: (data[Neighbor::Left.get_i(&self.geometry, i)]
                - data[Neighbor::Right.get_i(&self.geometry, i)])
                / (2.0 * dx),
            y: (data[Neighbor::Upper.get_i(&self.geometry, i)]
                - data[Neighbor::Bottom.get_i(&self.geometry, i)])
                / (2.0 * dx),
        }
    }

    #[inline]
    fn laplace(&self, i: usize, u: f64, dx: f64) -> f64 {
        (Neighbor::Left.get_v(self, i) - 2.0 * u
            + Neighbor::Right.get_v(self, i)
            + Neighbor::Upper.get_v(self, i)
            - 2.0 * u
            + Neighbor::Bottom.get_v(self, i))
            / (dx * dx)
    }

    pub fn display(&mut self, ui: &mut egui::Ui, color: bool) {
        let (mut min, mut max);
        if self.automax {
            (min, max) = self
                .data
                .iter()
                .copied()
                .minmax_by(f64::total_cmp)
                .into_option()
                .unwrap();
        } else {
            max = self.max; //if max < 1.0 {1.0} else {max};
            min = self.min; //if min > 0.0 {0.0} else {min};
        }
        let texture: &mut egui::TextureHandle = self.texture.get_or_insert_with(|| {
            // Load the texture only once.
            ui.ctx()
                .load_texture("my-image", egui::ColorImage::example(), Default::default())
        });
        if min.is_nan() || max.is_nan() || min > max {
            max = 1.0;
            min = -1.0;
        }
        let absmax = [min.abs(), max.abs()].iter().copied().max_by(f64::total_cmp).unwrap();
        let text: Vec<Color32> = self
            .data
            .iter().zip(&self.prev_data)
            .map(|(v, p)| {
                let g = ((v.clamp(min, max) - min) / (max - min + 0.001) * 255.0) as u8;
                if color {
                    let r = (((v - p) / absmax / self.dt * 1000.0 + 1.0).ln() * 15.0) as u8;
                    let b = (((p - v) / absmax / self.dt * 1000.0 + 1.0).ln() * 15.0) as u8;
                    Color32::from_rgb(r + b, g/5 + b/5, g)
                } else {
                    Color32::from_gray(g)
                }
            })
            .collect();
        texture.set(
            ColorImage {
                pixels: text,
                size: [self.geometry.width, self.geometry.height],
            },
            TextureOptions::NEAREST,
        );

        // Show the image:
        ui.add(Image::new((texture.id(), Vec2::new(self.geometry.width as f32, self.geometry.height as f32))).maintain_aspect_ratio(true).shrink_to_fit());
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

#[derive(Clone, Copy)]
struct Vect {
    x: f64,
    y: f64,
}

impl Vect {
    fn scal(self, other: Vect) -> f64 {
        self.x * other.x + self.y * other.y
    }
    fn square(self) -> f64 {
        self.scal(self)
    }
}
