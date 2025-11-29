from flask import Flask, render_template, redirect, url_for, flash, request, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from flask_migrate import Migrate
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
import os

# --- ReportLab Imports ---
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT, TA_JUSTIFY

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_change_this_in_production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

if not os.path.exists('static'):
    os.makedirs('static')

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

migrate = Migrate(app, db)

# --- Configuration & Constants ---
COMPANY_INFO = {
    "name": "AUTOASSESS AI SOLUTIONS",
    "address": "123 Tech Park, Innovation Way",
    "city": "Bangalore, KA 560001",
    "phone": "+91 98765 43210",
    "email": "support@autoassess.ai"
}

MODEL_PATHS = {
    "2wheeler": r'2_best.pt',
    "4wheeler": r'4_best.pt',
    "6wheeler": r'6_best.pt'
}

DAMAGE_DESCRIPTIONS_MAP = {
    'scratch': "Surface abrasion detected. Requires sanding, primer application, and color-matched repainting.",
    'dent': "Deformation of body panel. Requires stud welding/pulling and surface leveling.",
    'broken': "Severe structural failure. Complete part replacement is recommended to ensure safety.",
    'crack': "Visible fissure in material. Structural integrity compromised; replacement advised.",
    'shattered': "Glass/Plastic shattered. Immediate replacement required.",
    'default': "Damage detected by AI. Physical inspection recommended for final labor estimation."
}

PARTS_COST = {
    "2wheeler": {
        "Hero Splendor": {'broken': 50000, 'scratch': 1500, 'headlight': 2500, 'seat': 1500, 'tire': 800, 'mirror': 300},
        "Honda Activa": {'broken': 48000, 'scratch': 1400, 'headlight': 2300, 'seat': 1400, 'tire': 700, 'mirror': 250},
        "Honda Shine": {'broken': 49000, 'scratch': 1450, 'headlight': 2400, 'seat': 1450, 'tire': 750, 'mirror': 280},
        "Bajaj Pulsar": {'broken': 48000, 'scratch': 1400, 'headlight': 2300, 'seat': 1400, 'tire': 700, 'mirror': 250},
        "TVS Jupiter": {'broken': 49000, 'scratch': 1450, 'headlight': 2400, 'seat': 1450, 'tire': 750, 'mirror': 280}
    },
    "4wheeler": {
        "Maruti Suzuki: Swift": {'bumper': 8000, 'fender': 6000, 'front-windshield': 7000, 'rear-windshield': 7000, 'side-mirror': 1500, 'side-screen': 2000, 'door': 13000, 'headlamp': 2900, 'hood': 4000},
        "Tata Motors: Nexon": {'bumper': 8500, 'fender': 6200, 'front-windshield': 7100, 'rear-windshield': 7100, 'side-mirror': 1600, 'side-screen': 2100, 'door': 13500, 'headlamp': 3000, 'hood': 4200},
        "Hyundai: Creta": {'bumper': 9000, 'fender': 6500, 'front-windshield': 7200, 'rear-windshield': 7200, 'side-mirror': 1700, 'side-screen': 2200, 'door': 14000, 'headlamp': 3100, 'hood': 4400},
        "Mahindra: Scorpio": {'bumper': 9000, 'fender': 6500, 'front-windshield': 7200, 'rear-windshield': 7200, 'side-mirror': 1700, 'side-screen': 2200, 'door': 14000, 'headlamp': 3100, 'hood': 4400},
        "Toyota: Innova Crysta": {'bumper': 9000, 'fender': 6500, 'front-windshield': 7200, 'rear-windshield': 7200, 'side-mirror': 1700, 'side-screen': 2200, 'door': 14000, 'headlamp': 3100, 'hood': 4400}
    },
    "6wheeler": {
        "Bharat Benz 1923C Tipper": {'rear-lamp-l-damaged': 1000, 'rearlamp-r-damaged': 1000, 'sideboard-l-damaged': 1500, 'sideboard-r-damaged': 1500},
        "Tata LPT 1916": {'rear-lamp-l-damaged': 1200, 'rearlamp-r-damaged': 1200, 'sideboard-l-damaged': 1700, 'sideboard-r-damaged': 1700},
        "Tata Signa 1918.k": {'rear-lamp-l-damaged': 1100, 'rearlamp-r-damaged': 1100, 'sideboard-l-damaged': 1600, 'sideboard-r-damaged': 1600},
        "Mahindra Furio 16 Truck": {'rear-lamp-l-damaged': 1200, 'rearlamp-r-damaged': 1200, 'sideboard-l-damaged': 1700, 'sideboard-r-damaged': 1700},
        "Bharat Benz 1415RE Truck": {'rear-lamp-l-damaged': 1100, 'rearlamp-r-damaged': 1100, 'sideboard-l-damaged': 1600, 'sideboard-r-damaged': 1600}
    }
}

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(1), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)
    user_status = db.Column(db.String(10), default='Pending')
    vehicle_damages = db.relationship('VehicleDamage', backref='user', lazy=True)

class VehicleDamage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vehicle_type = db.Column(db.String(100), nullable=False)
    vehicle_brand = db.Column(db.String(100), nullable=False)
    detected_damage = db.Column(db.String(100), nullable=False)
    cost = db.Column(db.Float, nullable=False)
    total_cost = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def load_yolo_model(vehicle_type):
    model_path = MODEL_PATHS.get(vehicle_type)
    if model_path and os.path.exists(model_path):
        return YOLO(model_path)
    return None

def draw_bounding_boxes(image, results, class_names):
    color = (0, 0, 255)
    if not isinstance(image, np.ndarray):
        image = np.array(image)[:, :, ::-1]
    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls = int(cls)
        label = class_names[cls]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_width, label_height = label_size
        text_x = x1 + 5
        text_y = y1 + label_height + 5
        text_y = min(image.shape[0] - 5, max(text_y, 0))
        cv2.rectangle(image, (x1, y1 - label_height - 5),
                      (x1 + label_width + 10, y1), color, -1)
        cv2.putText(image, label, (text_x, text_y - label_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return image

def get_damage_description(damage_name):
    damage_name_lower = damage_name.lower()
    for key, description in DAMAGE_DESCRIPTIONS_MAP.items():
        if key in damage_name_lower:
            return description
    return DAMAGE_DESCRIPTIONS_MAP['default']

def get_image_dims(image_path, max_width, max_height):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            aspect = width / float(height)
            if width > max_width:
                width = max_width
                height = width / aspect
            if height > max_height:
                height = max_height
                width = height * aspect
            return width, height
    except:
        return max_width, max_height

# --- FIXED PDF GENERATOR ---
def generate_pdf(vehicle_type, brand, damage_details, total_cost, image_path):
    filename = f"Report_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf_save_path = os.path.join('static', filename)
    
    # 1. Setup Document
    doc = SimpleDocTemplate(
        pdf_save_path, 
        pagesize=letter,
        rightMargin=40, leftMargin=40, 
        topMargin=30, bottomMargin=30  # Reduced margins
    )
    elements = []
    styles = getSampleStyleSheet()
    
    # 2. Define Custom Styles (FIXED LEADING HERE)
    
    # Style for "AUTOASSESS AI SOLUTIONS"
    # IMPORTANT: leading (22) must be > fontSize (18) to prevent overlap
    s_company_name = ParagraphStyle(
        'CompanyName', 
        parent=styles['Normal'], 
        fontName='Helvetica-Bold', 
        fontSize=18, 
        leading=22,  # Fix: Increased leading to prevent overlap
        textColor=colors.HexColor('#003366'),
        spaceAfter=6  # Add space after the title
    )
    
    # Style for Address info
    s_company_details = ParagraphStyle(
        'CompanyDetails', 
        parent=styles['Normal'], 
        fontSize=9, 
        leading=12, 
        textColor=colors.HexColor('#555555')
    )
    
    s_report_title = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=16, alignment=TA_RIGHT, textColor=colors.HexColor('#2980b9'), spaceAfter=2)
    s_report_meta = ParagraphStyle('ReportMeta', parent=styles['Normal'], fontSize=9, leading=12, alignment=TA_RIGHT)
    s_normal = ParagraphStyle('MyBody', parent=styles['Normal'], fontSize=9, leading=11)
    
    # 3. Header Section (Two Columns)
    
    # Left Column Content (Company Info)
    # We pass these as a list of Paragraph objects
    company_info_content = [
        Paragraph(COMPANY_INFO['name'], s_company_name),
        Paragraph(COMPANY_INFO['address'], s_company_details),
        Paragraph(f"{COMPANY_INFO['city']} | {COMPANY_INFO['phone']}", s_company_details),
        Paragraph(COMPANY_INFO['email'], s_company_details),
    ]

    # Right Column Content (Report Meta)
    report_meta_content = [
        Paragraph("DAMAGE ASSESSMENT REPORT", s_report_title),
        Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d-%b-%Y')}", s_report_meta),
        Paragraph(f"<b>Report ID:</b> {filename.split('_')[1].split('.')[0]}", s_report_meta),
        Paragraph("<b>Generated by:</b> AI AutoAssess System", s_report_meta),
    ]
    
    # Create the Header Table
    # Note: We put the list of Paragraphs inside the cell list `[[ ... ]]`
    header_table = Table(
        [[company_info_content, report_meta_content]], 
        colWidths=[300, 230]
    )
    
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'), # Align everything to top
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    
    elements.append(header_table)
    elements.append(Spacer(1, 15))
    
    # Horizontal Divider
    elements.append(Table([['']], colWidths=[530], style=[('LINEBELOW', (0,0), (-1,-1), 1, colors.HexColor('#bdc3c7'))]))
    elements.append(Spacer(1, 15))

    # 4. Vehicle & User Info Strip
    v_data = [[
        Paragraph(f"<b>VEHICLE:</b><br/>{vehicle_type.upper()}", s_normal),
        Paragraph(f"<b>BRAND/MODEL:</b><br/>{brand.upper()}", s_normal),
        Paragraph(f"<b>CUSTOMER:</b><br/>{current_user.username.upper()}", s_normal)
    ]]
    
    t_vehicle = Table(v_data, colWidths=[175, 175, 180])
    t_vehicle.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0f2f5')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t_vehicle)
    elements.append(Spacer(1, 20))

    # 5. Damage Cost Table
    col_widths = [130, 290, 110]
    
    # Header Row
    tbl_header = [
        Paragraph("<b>COMPONENT</b>", ParagraphStyle('TH', parent=s_normal, textColor=colors.white)), 
        Paragraph("<b>ASSESSMENT NOTES</b>", ParagraphStyle('TH', parent=s_normal, textColor=colors.white)), 
        Paragraph("<b>COST (INR)</b>", ParagraphStyle('TH_R', parent=s_normal, textColor=colors.white, alignment=TA_RIGHT))
    ]
    
    table_data = [tbl_header]
    
    # Data Rows
    for damage in damage_details:
        part = damage['damage_type'].replace('-', ' ').title()
        desc = damage.get('description', 'Repair required.')
        cost = f"Rs. {damage['cost']:,.2f}"
        
        row = [
            Paragraph(part, s_normal),
            Paragraph(desc, s_normal),
            Paragraph(cost, ParagraphStyle('TC_R', parent=s_normal, alignment=TA_RIGHT))
        ]
        table_data.append(row)

    # Total Row
    total_row = [
        '', 
        Paragraph("<b>TOTAL ESTIMATE</b>", ParagraphStyle('TotalLabel', parent=s_normal, fontSize=10, alignment=TA_RIGHT)),
        Paragraph(f"<b>Rs. {total_cost:,.2f}</b>", ParagraphStyle('TotalVal', parent=s_normal, fontSize=10, alignment=TA_RIGHT, textColor=colors.HexColor('#c0392b')))
    ]
    table_data.append(total_row)

    t_cost = Table(table_data, colWidths=col_widths)
    t_cost.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')), # Header Blue
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -2), 0.5, colors.HexColor('#bdc3c7')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, -2), (-1, -2), 1, colors.black),
    ]))
    
    elements.append(t_cost)
    elements.append(Spacer(1, 20))

    # 6. Visual Evidence (Dynamically Sized)
    if os.path.exists(image_path):
        elements.append(Paragraph("<b>VISUAL EVIDENCE</b>", s_normal))
        elements.append(Spacer(1, 5))
        
        # Calculate fit within 500x220 points to ensure it fits on page
        w, h = get_image_dims(image_path, max_width=500, max_height=220)
        
        try:
            img = RLImage(image_path, width=w, height=h)
            img.hAlign = 'LEFT'
            elements.append(img)
        except:
            elements.append(Paragraph("[Image load error]", s_normal))

    # 7. Footer (Bottom of the content flow)
    elements.append(Spacer(1, 20))
    
    sig_table = Table([
        [Paragraph("__________________________", s_normal), Paragraph("", s_normal)],
        [Paragraph("Authorized Signature", s_normal), Paragraph("", s_normal)]
    ], colWidths=[200, 330])
    elements.append(sig_table)
    
    elements.append(Spacer(1, 10))
    
    disclaimer = ("<b>DISCLAIMER:</b> This is an AI-generated estimate. Actual repair costs may vary. "
                  "Please consult a certified service center.")
    elements.append(Paragraph(disclaimer, ParagraphStyle('Disc', parent=s_normal, fontSize=7, textColor=colors.grey)))

    doc.build(elements)
    return filename

# --- Routes ---
@app.context_processor
def inject_env():
    return {
        'GMAP_API_URL': os.environ.get('GMAP_API_KEY')
    }

@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email == 'admin@gmail.com' and password == 'admin':
            return redirect(url_for('admin'))
        user = User.query.filter_by(email=email, user_status='approved').first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('auth.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        age = request.form.get('age')
        gender = request.form.get('gender')
        mobile = request.form.get('mobile')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('auth.html')
            
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password, age=int(age), gender=gender, mobile=mobile)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Wait for approval.', 'success')
            return redirect(url_for('login'))
        except:
            db.session.rollback()
            flash('Registration failed.', 'danger')
    return render_template('auth.html')

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        vehicle_type = request.form.get('vehicle_type')
        brand = request.form.get('vehicle_brand')
        uploaded_file = request.files.get('image_file')

        if not uploaded_file or vehicle_type not in MODEL_PATHS or not brand:
            flash('Missing data.', 'danger')
            return render_template('upload.html')

        model_path = MODEL_PATHS.get(vehicle_type)
        if not os.path.exists(model_path):
            flash('Model not found.', 'danger')
            return render_template('upload.html')

        try:
            model = load_yolo_model(vehicle_type)
            image = Image.open(uploaded_file.stream)
            results = model(image)
            
            if not results or len(results[0].boxes) == 0:
                flash('No damage detected.', 'info')
                return render_template('upload.html')

            result = results[0]
            class_names = result.names
            detected_classes = [int(cls) for cls in result.boxes.cls]
            
            damage_details = []
            for detected_class in detected_classes:
                original_damage_name = class_names[detected_class]
                normalized = original_damage_name.lower().replace(' ', '')
                while '--' in normalized: normalized = normalized.replace('--', '-')
                
                cost = PARTS_COST.get(vehicle_type, {}).get(brand, {}).get(normalized, 0)
                description = get_damage_description(original_damage_name)
                
                damage_details.append({
                    'damage_type': original_damage_name, 
                    'cost': cost,
                    'description': description
                })

            total_cost = sum(d['cost'] for d in damage_details)
            image_with_boxes = draw_bounding_boxes(np.array(image), results, class_names)
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            image_filename = f"output_{timestamp}.jpg"
            cv2.imwrite(os.path.join('static', image_filename), image_with_boxes)

            # DB Storage
            for damage in damage_details:
                db.session.add(VehicleDamage(
                    vehicle_type=vehicle_type,
                    vehicle_brand=brand,
                    detected_damage=damage['damage_type'],
                    cost=damage['cost'],
                    total_cost=total_cost,
                    image_path=image_filename,
                    user_id=current_user.id
                ))
            db.session.commit()

            pdf_filename = generate_pdf(vehicle_type, brand, damage_details, total_cost, os.path.join('static', image_filename))

            flash('Assessment complete.', 'success')
            return render_template('upload.html', image_filename=image_filename, repair_cost=total_cost, damage_details=damage_details, pdf_filename=pdf_filename)

        except Exception as e:
            flash('Error processing image.', 'danger')
            print(e)
            return render_template('upload.html')

    return render_template('upload.html')

@app.route('/assessment')
@login_required
def assessment():
    all_damages = VehicleDamage.query.filter_by(user_id=current_user.id).order_by(VehicleDamage.created_at.desc()).all()
    grouped_assessments = {}
    for damage in all_damages:
        key = damage.image_path
        if key not in grouped_assessments:
            grouped_assessments[key] = {
                'vehicle_brand': damage.vehicle_brand,
                'vehicle_type': damage.vehicle_type,
                'total_cost': damage.total_cost,
                'image_path': damage.image_path,
                'created_at': damage.created_at,
                'damages': [] 
            }
        grouped_assessments[key]['damages'].append({
            'detected_damage': damage.detected_damage,
            'cost': damage.cost
        })
    return render_template('assessment.html', results=list(grouped_assessments.values()))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin')
def admin(): return render_template('admin.html')

@app.route('/view')
def view(): return render_template('view.html', data=User.query.all())

@app.route('/delete/<int:id>/', methods=['POST'])
def delete_user(id):
    u = User.query.get(id)
    if u:
        VehicleDamage.query.filter_by(user_id=id).delete()
        db.session.delete(u)
        db.session.commit()
    return redirect(url_for('view'))

@app.route('/view_requests')
def view_requests(): return render_template('view_requests.html', data=User.query.filter_by(user_status='Pending').all())

@app.route('/update_status/<int:id>/')
def update_status(id):
    User.query.get(id).user_status = 'approved'
    db.session.commit()
    return redirect(url_for('view_requests'))

@app.route('/metrics')
def metrics(): return render_template('metrics.html')

@app.context_processor
def inject_user(): return dict(current_user=current_user)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)